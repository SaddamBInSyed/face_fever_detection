/*
 * Copyright (C) Ogal Optronics Systems Ltd.
 *
 * Author: Sebastian Cabot
 *
 *
 * Your desired license
 *  
 */

#include <cstdlib>
#include <cstdio>
#include <cerrno>

#include <atomic>
#include <vector>

#include "op.h"

extern "C"
{
#include <signal.h>
#include <argp.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
}

namespace
{
    std::atomic<bool> g_stop(false);
    std::string g_ipcNmae;
    int g_log2Length = 2;
    bool g_dumpFiles = false;
    void sig_handler(int)
    {
        g_stop = true;
    }
}

static int parse_opt(int key, char *arg, struct argp_state *state)
{
    switch (key)
    {
        case 'd':
        {
            g_dumpFiles = true;
        }
        break;
        case 'l':
        {
            try
            {
                g_log2Length = std::stoi(arg);
                if (g_log2Length < 0)
                {
                    fprintf(stderr,
                            "Log 2 Length of %d is less than 0. 0 will be used\n",
                            g_log2Length);
                    g_log2Length = 0;
                }
                if (g_log2Length > 4)
                {
                    fprintf(stderr,
                            "Log 2 Length of %d is greater than 4. 4 will be used\n",
                            g_log2Length);
                    g_log2Length = 4;
                }
            }
            catch (...)
            {
                argp_error(state,
                           "Invalid LOG2 Length. The input should be a number between 0 and 4");
            }
        }
        break;
        case ARGP_KEY_ARG:
            if (state->arg_num == 0)
            {
                g_ipcNmae = arg;
            }
        break;
        case ARGP_KEY_END:
            if (state->arg_num < 1)
            {
                argp_error(state,
                           "Please specify SHARED_NAME for the ipc buffers.");
            }
            if (state->arg_num > 1)
            {
                argp_error(state,
                           "Please specify just one SHARED_NAME for the ipc buffers.");
            }
        break;
    }
    return 0;
}

int main(int argc, char * argv[])
{
    static constexpr uint32_t memorySize = 384U * 288U * 2U + 64U; // Allows storing a full frame of 16bit values + header

    struct sigaction sa = {
                            0 };
    sa.sa_handler = sig_handler;
    sigfillset(&sa.sa_mask);

    sigaction(SIGINT,
              &sa,
              NULL);

    struct argp_option options[] = {
                                     {
                                       "log2length",
                                       'l',
                                       "LOG2",
                                       0,
                                       "The log 2 of number of ipc buffers to allocate.\nValid values are in the range of 0 for one buffer and 4 for 16 buffers.\nThe default is 2." },
                                     {
                                       0,
                                       'd',
                                       0,
                                       0,
                                       "Specify to dump the incoming frames to files" },
                                     {
                                       0 } };
    struct argp argp = {
                         options,
                         parse_opt,
                         "SHARED_NAME" };

    argp_parse(&argp,
               argc,
               argv,
               0,
               NULL,
               0);

    op::IPC_cap_client client;


    if (client.initialize(g_ipcNmae,
                          uint32_t(1 << g_log2Length),
                          memorySize))
    {
        client.sync(); // Make sure we start at a known time
        do
        {
            uint8_t* pImage = static_cast<uint8_t*>(client.lock_next_image_for_read());
            if(nullptr != pImage)
            {
                uint16_t *imageHeader = (uint16_t*) pImage;
                pImage += 64; // Skip header
                // Uncomment next line to get the sensor's serial number
                // uint32_t serialNumber = imageHeader[5] | (uint32_t(imageHeader[6]) << 16);
                uint16_t imageHeight = imageHeader[11];
                uint16_t imageWidth = imageHeader[12];
                uint32_t imageId = imageHeader[26] | (uint32_t(imageHeader[27]) << 16);

                printf("Got frame %u just as you wanted with dimensions (%d,%d)\n",
                       imageId,
                       imageWidth,
                       imageHeight);
                // We have an image in our hands now
		
		// Send callback


		// Dump to disk
		if (g_dumpFiles)
                {
                    printf("saved image like you want\n");
                    std::string sFilename = R"(/opt/eyerop/dump/frame_)"
                                            + std::to_string(imageId) + R"(.bin)";
                    int fd = ::open(sFilename.c_str(),
                                    O_CREAT | O_WRONLY,
                                    S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
                    if (0 <= fd)
                    {
                        ::write(fd,
                                imageHeader,
                                64);
                        ::write(fd,
                                pImage,
                                imageHeight * imageWidth * 2);
                        ::close(fd);
                    } // NO NEED FOR ELSE
                } // NO NEED FOR ELSE
                // Make sure we release the locked image
                client.release_locked_image();
            }// NO NEED FOR ELSE
        }while(!g_stop);
    }
    else
    {
        fprintf(stderr,
                "Unable to create IPC client\n");
    }


    return 0;
}

