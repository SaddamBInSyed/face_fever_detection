/*
 * op.h
 *
 *  Created on: Oct 13, 2015
 *      Author: sebastian
 */

#ifndef OP_H_
#define OP_H_

#include <string>
#include <memory>
#include <vector>

#if !defined(DISALLOW_COPY_AND_MOVE)
#    define DISALLOW_COPY(type) \
        type(const type &) = delete; \
        type & operator =(const type &) = delete;
#    define DISALLOW_MOVE(type) \
        type(type &&) = delete; \
        type & operator =(type &&) = delete;
#    define DISALLOW_COPY_AND_MOVE(type) \
        DISALLOW_COPY(type) \
        DISALLOW_MOVE(type)
#endif /* DISALLOW_COPY */

extern "C"
{
    void *op_aligned_alloc(uint32_t alignment, uint32_t size);
    void *op_malloc(uint32_t size);
    void *op_realloc(void *ptr, uint32_t size);
    void *op_calloc(uint32_t nmemb, uint32_t size);
    void op_free(void *ptr);
    const char* libop_version();
}

namespace op
{
    class OPmanaged
    {
    public:
        virtual ~OPmanaged() = default;

        static void * operator new(std::size_t sz) noexcept
        {
            return op_malloc(sz);
        }
        static void * operator new[](std::size_t sz) noexcept
        {
            return op_malloc(sz);
        }
        static void operator delete(void *ptr) noexcept
        {
            op_free(ptr);
        }
        static void operator delete[](void *ptr) noexcept
        {
            op_free(ptr);
        }

    protected:
        OPmanaged() = default;

    };

    class Waitable_object: public OPmanaged
    {
    public:
        virtual ~Waitable_object() = default;

        virtual bool post() = 0;
        virtual bool try_wait() = 0;
        virtual bool wait() = 0;
        virtual bool wait(uint32_t timeoutus) = 0;

    protected:
        Waitable_object() = default;
    };

    /** Kernel backed shared IPC auto reset event implemented using semaphores
     * An auto reset event will signal just one waiting process and reset itself immediately once read.
     * It is in effect a mutex.
     */
    class Shared_auto_reset_event: public Waitable_object
    {
    public:
        typedef std::shared_ptr<Shared_auto_reset_event> Ptr;

        DISALLOW_COPY_AND_MOVE (Shared_auto_reset_event)

        ~Shared_auto_reset_event() = default;

        static Ptr open(const std::string &name,
                        bool failIfExists,
                        bool state,
                        bool master = false);

        bool post() override;
        bool try_wait() override;
        bool wait() override;
        bool wait(uint32_t timeoutus) override;

    private:
        struct Shared_auto_reset_event_state;
        Shared_auto_reset_event() = default;

        std::unique_ptr<Shared_auto_reset_event_state> state_;
    };

    class Shared_memory_object: public OPmanaged
    {
    public:

        typedef std::shared_ptr<Shared_memory_object> Ptr;

        DISALLOW_COPY_AND_MOVE (Shared_memory_object)

        ~Shared_memory_object() = default;

        //! Create or open a shared memory region
        //! If the memory exists and the requested size is bigger than the already allocated size then the call will fail
        //! The actually allocated size will be returned in size
        static Ptr open(const std::string & name,
                        bool failIfExists,
                        std::size_t & size,
                        bool master = false);

        bool wait_for_data(uint32_t timeoutus);

        void* data() const;

        void* data_lock_write();
        void* data_try_lock_write();
        void data_unlock_write(bool setReady);

        void* data_lock_read();
        void* data_try_lock_read();
        void data_unlock_read();
        //! return the valid allocated size of the shared memory
        std::size_t alloc_size() const;

    private:
        struct Shared_memory_state;
        std::unique_ptr<Shared_memory_state> state_;
        Shared_memory_object();

    };

    //! This class is used to hold IPC integer value.
    //! All access to the underlying value is done atomically
    class Shared_atomic_int_value: public OPmanaged
    {
    public:
        typedef std::shared_ptr<Shared_atomic_int_value> Ptr;

        DISALLOW_COPY_AND_MOVE (Shared_atomic_int_value)

        ~Shared_atomic_int_value() = default;
        //! Create or open a shared int value (The size is dependent on the architecture)
        static Ptr open(const std::string & name, bool failIfExists, bool master = false);

        intptr_t get_value();
        void set_value(intptr_t v);

        bool wait_for_data(uint32_t timeoutus);

    private:
        Shared_atomic_int_value() = default;
        Shared_memory_object::Ptr value_;
    };

    class IPC_cap_client
    {
    public:

        IPC_cap_client() = default;

        ~IPC_cap_client() = default;

        bool initialize(std::string const &baseName,
                        uint32_t sharedImageCount,
                        std::size_t expectedImageSize);

        bool sync();

        void* lock_next_image_for_read();

        void release_locked_image();

    private:
        std::vector<Shared_memory_object::Ptr> sharedImages_;
        Shared_atomic_int_value::Ptr lastImageWritten_;
        std::vector<Shared_memory_object::Ptr>::iterator lockedImage_;
    };

}    //namespace op

#endif /* OP_H_ */
