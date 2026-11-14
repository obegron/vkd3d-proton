/*
 * Copyright 2026 vkd3d-proton contributors
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API

#include "vkd3d_private.h"

/* Workaround for applications which read a READBACK buffer on the CPU
 * without waiting for the GPU copy that writes it.
 *
 * Death Stranding 2 copies a GPU visibility list into a READBACK buffer every frame
 * and reads it in place. Before reading, it waits for the last fence value submitted
 * to the queue rather than for the submission that contains the copy. If the copy
 * has been recorded but not submitted yet, the wait completes immediately, and the
 * CPU reads the buffer while the GPU overwrites it.
 *
 * The game maps the buffer before every read, so Map() can restore the missing
 * dependency. We track every copy into a READBACK buffer from recording, through
 * ExecuteCommandLists(), to the first Signal() on that queue after it.
 * Map() then waits until all copies into the buffer that were recorded before the
 * Map() call have completed.
 *
 * The wait is bounded. An application could legitimately map a buffer before it
 * submits a copy into it on the same thread, and we must not deadlock in that case. */

#define VKD3D_READBACK_SYNC_TIMEOUT_NS (4ull * 1000 * 1000)

static bool d3d12_resource_is_readback_buffer(const struct d3d12_resource *resource)
{
    const D3D12_HEAP_PROPERTIES *props = &resource->heap_properties;

    if (!d3d12_resource_is_buffer(resource) || (resource->flags & VKD3D_RESOURCE_RESERVED))
        return false;

    return props->Type == D3D12_HEAP_TYPE_READBACK ||
            (props->Type == D3D12_HEAP_TYPE_CUSTOM && props->CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK);
}

HRESULT vkd3d_readback_sync_init(struct vkd3d_readback_sync *sync)
{
    int rc;

    memset(sync, 0, sizeof(*sync));

    if ((rc = pthread_mutex_init(&sync->mutex, NULL)))
        return hresult_from_errno(rc);

    if ((rc = condvar_reltime_init(&sync->cond)))
    {
        pthread_mutex_destroy(&sync->mutex);
        return hresult_from_errno(rc);
    }

    return S_OK;
}

void vkd3d_readback_sync_cleanup(struct vkd3d_readback_sync *sync)
{
    size_t i;

    for (i = 0; i < sync->submission_count; i++)
        if (sync->submissions[i].fence)
            d3d12_fence_iface_dec_ref((d3d12_fence_iface *)sync->submissions[i].fence);

    for (i = 0; i < sync->stale_event_count; i++)
        vkd3d_native_sync_handle_destroy(sync->stale_events[i]);

    vkd3d_free(sync->copies);
    vkd3d_free(sync->submissions);
    vkd3d_free(sync->stale_events);
    condvar_reltime_destroy(&sync->cond);
    pthread_mutex_destroy(&sync->mutex);
}

static void vkd3d_readback_sync_remove_submission_locked(struct vkd3d_readback_sync *sync, size_t index)
{
    if (sync->submissions[index].fence)
        d3d12_fence_iface_dec_ref((d3d12_fence_iface *)sync->submissions[index].fence);
    sync->submissions[index] = sync->submissions[--sync->submission_count];
}

static void vkd3d_readback_sync_prune_locked(struct vkd3d_readback_sync *sync)
{
    struct vkd3d_readback_sync_submission *submission;
    size_t i;

    /* Applications which keep a buffer persistently mapped never reach the wait in Map(),
     * so completed submissions must also be dropped here. */
    for (i = 0; i < sync->submission_count; )
    {
        submission = &sync->submissions[i];
        if (submission->fence && ID3D12Fence_GetCompletedValue(submission->fence) >= submission->value)
            vkd3d_readback_sync_remove_submission_locked(sync, i);
        else
            i++;
    }
}

void vkd3d_readback_sync_record_copy(struct vkd3d_readback_sync *sync,
        ID3D12CommandList *list, const struct d3d12_resource *dst)
{
    struct vkd3d_readback_sync_copy *copy;
    size_t i;

    if (!d3d12_resource_is_readback_buffer(dst))
        return;

    pthread_mutex_lock(&sync->mutex);

    /* One entry per buffer and command list is enough, since a list is always submitted as a whole. */
    for (i = 0; i < sync->copy_count; i++)
    {
        if (sync->copies[i].list == list && sync->copies[i].resource == dst)
        {
            pthread_mutex_unlock(&sync->mutex);
            return;
        }
    }

    if (vkd3d_array_reserve((void **)&sync->copies, &sync->copies_size,
            sync->copy_count + 1, sizeof(*sync->copies)))
    {
        copy = &sync->copies[sync->copy_count++];
        copy->list = list;
        copy->resource = dst;
        copy->sequence = ++sync->sequence;
        copy->submitted = false;
    }

    pthread_mutex_unlock(&sync->mutex);
}

void vkd3d_readback_sync_forget_list(struct vkd3d_readback_sync *sync, ID3D12CommandList *list)
{
    bool removed = false;
    size_t i;

    pthread_mutex_lock(&sync->mutex);

    /* Copies that were already submitted remain tracked as submissions. */
    for (i = 0; i < sync->copy_count; )
    {
        if (sync->copies[i].list == list)
        {
            sync->copies[i] = sync->copies[--sync->copy_count];
            removed = true;
        }
        else
            i++;
    }

    if (removed)
        condvar_reltime_broadcast(&sync->cond);
    pthread_mutex_unlock(&sync->mutex);
}

void vkd3d_readback_sync_forget_resource(struct vkd3d_readback_sync *sync, const struct d3d12_resource *resource)
{
    size_t i;

    if (!d3d12_resource_is_readback_buffer(resource))
        return;

    pthread_mutex_lock(&sync->mutex);

    for (i = 0; i < sync->copy_count; )
    {
        if (sync->copies[i].resource == resource)
            sync->copies[i] = sync->copies[--sync->copy_count];
        else
            i++;
    }

    for (i = 0; i < sync->submission_count; )
    {
        if (sync->submissions[i].resource == resource)
            vkd3d_readback_sync_remove_submission_locked(sync, i);
        else
            i++;
    }

    pthread_mutex_unlock(&sync->mutex);
}

void vkd3d_readback_sync_submit(struct vkd3d_readback_sync *sync, const struct d3d12_command_queue *queue,
        ID3D12CommandList * const *command_lists, UINT command_list_count)
{
    struct vkd3d_readback_sync_submission *submission;
    bool added = false;
    size_t i;
    UINT j;

    pthread_mutex_lock(&sync->mutex);

    vkd3d_readback_sync_prune_locked(sync);

    for (j = 0; j < command_list_count; j++)
    {
        for (i = 0; i < sync->copy_count; i++)
        {
            if (sync->copies[i].list != command_lists[j])
                continue;

            /* A closed list can be executed more than once, and every execution must be waited for.
             * Keep the sequence of the recorded copy, since that is what Map() compares against. */
            if (!vkd3d_array_reserve((void **)&sync->submissions, &sync->submissions_size,
                    sync->submission_count + 1, sizeof(*sync->submissions)))
                continue;

            submission = &sync->submissions[sync->submission_count++];
            submission->resource = sync->copies[i].resource;
            submission->queue = queue;
            submission->fence = NULL;
            submission->value = 0;
            submission->sequence = sync->copies[i].sequence;
            sync->copies[i].submitted = true;
            added = true;
        }
    }

    if (added)
        condvar_reltime_broadcast(&sync->cond);
    pthread_mutex_unlock(&sync->mutex);
}

void vkd3d_readback_sync_signal(struct vkd3d_readback_sync *sync, const struct d3d12_command_queue *queue,
        ID3D12Fence *fence, UINT64 value)
{
    bool updated = false;
    size_t i;

    pthread_mutex_lock(&sync->mutex);

    /* Signals on a queue complete in order, so the first signal after a submission
     * completes only once the submission itself has completed. */
    for (i = 0; i < sync->submission_count; i++)
    {
        if (sync->submissions[i].queue == queue && !sync->submissions[i].fence)
        {
            /* Use an internal reference. A public reference would keep the device alive. */
            d3d12_fence_iface_inc_ref((d3d12_fence_iface *)fence);
            sync->submissions[i].fence = fence;
            sync->submissions[i].value = value;
            updated = true;
        }
    }

    if (updated)
        condvar_reltime_broadcast(&sync->cond);
    pthread_mutex_unlock(&sync->mutex);
}

static bool vkd3d_readback_sync_wait_fence_locked(struct vkd3d_readback_sync *sync,
        ID3D12Fence *fence, uint64_t value, uint64_t timeout_ns)
{
    vkd3d_native_sync_handle event;
    bool signaled;

    if (FAILED(vkd3d_native_sync_handle_create(0, VKD3D_NATIVE_SYNC_HANDLE_TYPE_EVENT, &event)))
        return false;

    if (FAILED(d3d12_fence_iface_set_native_sync_handle_on_completion_explicit(fence,
            VKD3D_WAITING_EVENT_SINGLE, value, event, NULL)))
    {
        vkd3d_native_sync_handle_destroy(event);
        return false;
    }

    pthread_mutex_unlock(&sync->mutex);
    signaled = vkd3d_native_sync_handle_acquire_timeout(event, (timeout_ns + 999999) / 1000000);
    pthread_mutex_lock(&sync->mutex);

    if (signaled)
        vkd3d_native_sync_handle_destroy(event);
    else if (vkd3d_array_reserve((void **)&sync->stale_events, &sync->stale_events_size,
            sync->stale_event_count + 1, sizeof(*sync->stale_events)))
    {
        /* The fence still holds on to the event and will signal it eventually,
         * so it can only be destroyed together with the device. */
        sync->stale_events[sync->stale_event_count++] = event;
    }

    return signaled;
}

void vkd3d_readback_sync_wait(struct vkd3d_readback_sync *sync, const struct d3d12_resource *resource)
{
    uint64_t start_ns, elapsed_ns, sequence, value = 0;
    bool pending, waited = false;
    ID3D12Fence *fence;
    size_t i;

    if (!d3d12_resource_is_readback_buffer(resource))
        return;

    pthread_mutex_lock(&sync->mutex);

    /* Only wait for copies recorded before Map() was called, not for later frames. */
    sequence = sync->sequence;
    start_ns = vkd3d_get_current_time_ns();

    for (;;)
    {
        elapsed_ns = vkd3d_get_current_time_ns() - start_ns;
        if (elapsed_ns >= VKD3D_READBACK_SYNC_TIMEOUT_NS)
        {
            FIXME_ONCE("Timed out waiting for pending copy into readback buffer %p.\n", resource);
            break;
        }

        /* Recorded, but not submitted yet. */
        pending = false;
        for (i = 0; i < sync->copy_count && !pending; i++)
        {
            pending = sync->copies[i].resource == resource && !sync->copies[i].submitted &&
                    sync->copies[i].sequence <= sequence;
        }

        /* Submitted, but not followed by a signal yet. */
        fence = NULL;
        for (i = 0; i < sync->submission_count && !pending && !fence; i++)
        {
            if (sync->submissions[i].resource != resource || sync->submissions[i].sequence > sequence)
                continue;

            if (!sync->submissions[i].fence)
                pending = true;
            else if (ID3D12Fence_GetCompletedValue(sync->submissions[i].fence) >= sync->submissions[i].value)
                vkd3d_readback_sync_remove_submission_locked(sync, i--);
            else
            {
                fence = sync->submissions[i].fence;
                value = sync->submissions[i].value;
            }
        }

        if (pending)
        {
            condvar_reltime_wait_timeout_ns(&sync->cond, &sync->mutex, VKD3D_READBACK_SYNC_TIMEOUT_NS - elapsed_ns);
            waited = true;
        }
        else if (fence)
        {
            d3d12_fence_iface_inc_ref((d3d12_fence_iface *)fence);
            vkd3d_readback_sync_wait_fence_locked(sync, fence, value, VKD3D_READBACK_SYNC_TIMEOUT_NS - elapsed_ns);
            d3d12_fence_iface_dec_ref((d3d12_fence_iface *)fence);
            waited = true;
        }
        else
            break;
    }

    pthread_mutex_unlock(&sync->mutex);

    if (waited)
        TRACE("Waited %"PRIu64" us for pending copies into readback buffer %p.\n",
                (vkd3d_get_current_time_ns() - start_ns) / 1000, resource);
}
