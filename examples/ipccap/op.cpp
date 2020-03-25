/*
 * op.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: sebastian
 */

#include "op.h"

#include <cstdlib>
#include <cstdio>
#include <cerrno>

bool op::IPC_cap_client::initialize(const std::string& baseName,
                                    uint32_t sharedImageCount,
                                    std::size_t expectedImageSize)
{
    bool ret = false;
    std::string baseIpcName = baseName + R"(_omem)";
    fprintf(stdout, "op::IPC_cap_client::initialize baseIpcName = %s\n", baseIpcName.c_str());
    Shared_auto_reset_event::Ptr ipcInitDone = op::Shared_auto_reset_event::open(baseIpcName
                                                                                 + "init",
                                                                                 false,
                                                                                 false);
    if (ipcInitDone == NULL)
    {
        fprintf(stderr, "ERROR: op::IPC_cap_client::initialize FAILED : ipcInitDone is NULL ***maybe you should use sudo?***\n");
        return false;
    }
    
    if (ipcInitDone->wait())
    {
        // Server created all objects - great now it is our turn to open them
        lastImageWritten_ = Shared_atomic_int_value::open(baseIpcName,
                                                          false,
                                                          false);
        if(nullptr != lastImageWritten_)
        {
            for (uint32_t i = 0; i < sharedImageCount; ++i)
            {
                std::size_t requiredImageSize = expectedImageSize;
                auto shImage = op::Shared_memory_object::open(baseIpcName + std::to_string(i),
                                                              false,
                                                              requiredImageSize);
                if (nullptr != shImage && expectedImageSize <= requiredImageSize)
                {
                    sharedImages_.push_back(std::move(shImage));
                }
                else
                {
                    fprintf(stderr,
                            "Unable to create shared memory region\n");
                    break; // Bail out
                } // NO NEED FOR ELSE
            }
        }// NO NEED FOR ELSE
    } // NO NEED FOR ELSE

    ret = (sharedImages_.size() == sharedImageCount);

    return ret;
}

bool op::IPC_cap_client::sync()
{
    // We want to skip the first available image so as not to incur latency in case we start reading too late
    if(nullptr != lastImageWritten_)
    {
        return lastImageWritten_->wait_for_data(30000000); // Wait up to 30secs
    }// NO NEED FOR ELSE
    return false;
}

void* op::IPC_cap_client::lock_next_image_for_read()
{
    void* ret = nullptr;
    if(nullptr != lastImageWritten_)
    {
        if(lastImageWritten_->wait_for_data(10000)) // Wait up to 10ms
        {
            auto imageIndex = static_cast<uint32_t>(lastImageWritten_->get_value());
            if(imageIndex < sharedImages_.size())
            {
                // Lock for reading - this will ensure the buffer is not overwritten while we are reading from it
                ret = sharedImages_[imageIndex]->data_lock_read();
                lockedImage_ = sharedImages_.begin() + imageIndex;
            }// NO NEED FOR ELSE
        }// NO NEED FOR ELSE
    }// NO NEED FOR ELSE
    return ret;
}

void op::IPC_cap_client::release_locked_image()
{
    if(sharedImages_.end() != lockedImage_)
    {
        (*lockedImage_)->data_unlock_read();
        lockedImage_ = sharedImages_.end();
    }// NO NEED FOR ELSE
}
