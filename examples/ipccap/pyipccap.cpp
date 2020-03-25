/*
 * Copyright (C) Ogal Optronics Systems Ltd.
 *
 * Author: Andrey Berger
 *
 *
 * Your desired license
 *  
 */

/**************************************************************************************/
//includes
#include <iostream>
#include <python2.7/Python.h>
#include <python2.7/structmember.h>

#include <cstdlib>
#include <cstdio>
#include <cerrno>

#include <atomic>
#include <vector>

#include "op.h"

extern "C"
{

#include <argp.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
}

/**************************************************************************************/

// defines
/**************************************************************************************/

// debug prints
//#define PYIPCCAP_DEBUG
#ifdef PYIPCCAP_DEBUG
#define DPRINTF(fmt, args...) fprintf(stderr, "+++ " fmt, ## args)
#else
#define DPRINTF(fmt, args...)
#endif

#define ERRPRINTF(fmt, args...) fprintf(stderr, "+++ " fmt, ## args)

namespace
{
    /**************************************************************************************/
// private variables
    /**************************************************************************************/

    constexpr uint32_t memorySize = 384U * 288U * 2U + 64U; // Allows storing a full frame of 16bit values + header

    /* Struct to contain an IPC object name which can be None */
    typedef struct Object_name
    {
        bool isNone;
        std::string name;
    } t_object_name;

    /* internal cpp Struct to contain ipccap values that used in open and get data*/
    typedef struct Ipccap
    {
        std::string ipcNmae;
        op::IPC_cap_client client;
        int nOutputImageQ;
    } t_ipccap;

    /* Struct to contain an Python object */
    typedef struct Thermapp
    {
        PyObject_HEAD
        uintptr_t ipcCapHandle;
        uint16_t imageHeight = 0;
        uint16_t imageWidth = 0;
        uint32_t imageId = 0;
        uint32_t serialNumber = 0;
    } t_thermapp;

    /**************************************************************************************/
// private functions declaration
    /**************************************************************************************/

    /** @brief converts python string to C string
     *
     *  a callback referred by in PyArg_ParseTupleAndKeywords function
     *  Verifies that the py_name_param is either None or a string.
     *  If it's a string, checked_name->name points to a PyMalloc-ed buffer
     *  holding a NULL-terminated C version of the string when this function
     *  concludes. The caller is responsible for releasing the buffer.
     *
     *
     *  @param py_name_param object passed by python.
     *  @param checked_name object name c string.
     *  @return 1 in case os success (even if the name is none).
     *          0 in case of error
     */

    int convert_name_param(PyObject *pyNameParam, void *checkedName);

    /** @brief implementation of a the module from python
     *
     *  allocates the module
     *
     *  @return the python object
     */

    PyObject * thermapp_new(PyTypeObject *type, PyObject *args, PyObject *kwlist);

    /** @brief initilazation of a the module from python
     *
     *  runs after thermapp_new,
     *  performs parameters' parsing and assignment
     *
     *  @return 0 in case os success (even if the name is none).
     *         -1 in case of error
     */

    int thermapp_init(t_thermapp *self, PyObject *args, PyObject *keywords);

    /** @brief python module distructor implementation
     *
     *  @return void
     */

    void thermapp_dealloc(t_thermapp *self);

    /** @brief and initializes  opens client object
     *
     *  Function belongs to thermapp module
     *
     *
     *  @return python thermapp object
     */

    PyObject * pyipccap_open_client(t_thermapp *self, PyObject *args);

    /** @brief  tries to get image from shared memory region
     *
     *  Function belongs to thermapp module
     *  tries to return image buffer from shared memory
     *  if not exusuts, return NONE python object
     *
     *  @return python string array object in case os success\
 *          NONE objecy in case of not avialible
     */

    PyObject * pyipccap_get_data(t_thermapp *self, PyObject *args);

    // thermapp object functions list
    PyMethodDef thermappMethods[] = {
                                      {
                                        "open_shared_memory",
                                        (PyCFunction) pyipccap_open_client,
                                        METH_VARARGS,
                                        "open shared memory" },
                                      {
                                        "get_data",
                                        (PyCFunction) pyipccap_get_data,
                                        METH_VARARGS,
                                        "get data from shared object" },

                                      {
                                        nullptr } /* Sentinel */
    };

    // thermapp object members list
    PyMemberDef thermappMembers[] = {
                                      {
                                        (char *) "Images",
                                        T_ULONG,
                                        offsetof(t_thermapp,
                                                 ipcCapHandle),
                                        READONLY,
                                        (char *) "pointer to shared memory data" },

                                      {
                                        (char *) "imageHeight",

                                        T_USHORT,
                                        offsetof(t_thermapp,
                                                 imageHeight),
                                        READONLY,
                                        (char *) " Image Height" },

                                      {
                                        (char *) "imageWidth",

                                        T_USHORT,
                                        offsetof(t_thermapp,
                                                 imageWidth),
                                        READONLY,
                                        (char *) "Image Width" },

                                      {
                                        (char *) "imageId",

                                        T_UINT,
                                        offsetof(t_thermapp,
                                                 imageId),
                                        READONLY,
                                        (char *) "Image ID" },

                                      {
                                        (char *) "serialNumber",

                                        T_UINT,
                                        offsetof(t_thermapp,
                                                 serialNumber),
                                        READONLY,
                                        (char *) "Image serial Number" },

                                      {
                                        nullptr } /* Sentinel */

    };

    // thermapp object fields
    PyTypeObject thermappType = {
                                  PyVarObject_HEAD_INIT(nullptr,
                                                        0) "pyipccap.thermapp", // tp_name
                                  sizeof(t_thermapp),                  // tp_basicsize
                                  0,                                  // tp_itemsize
                                  (destructor) thermapp_dealloc,     // tp_dealloc
                                  0,                                  // tp_print
                                  0,                                  // tp_getattr
                                  0,                                  // tp_setattr
                                  0,                                  // tp_compare
                                  0,                                  // tp_repr
                                  0,                                  // tp_as_number
                                  0,                                  // tp_as_sequence
                                  0,                                  // tp_as_mapping
                                  0,                                  // tp_hash
                                  0,                                  // tp_call
                                  0,                                  // tp_str
                                  0,                                  // tp_getattro
                                  0,                                  // tp_setattro
                                  0,                                  // tp_as_buffer
                                  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   // tp_flags
                                  "thermapp object",                  // tp_doc
                                  0,                                  // tp_traverse
                                  0,                                  // tp_clear
                                  0,                                  // tp_richcompare
                                  0,                                 // tp_weaklistoffset
                                  0,                                  // tp_iter
                                  0,                                  // tp_iternext
                                  thermappMethods,                  // tp_methods
                                  thermappMembers,                  // tp_members
                                  0,                // tp_getset
                                  0,                                  // tp_base
                                  0,                                  // tp_dict
                                  0,                                  // tp_descr_get
                                  0,                                  // tp_descr_set
                                  0,                                  // tp_dictoffset
                                  (initproc) thermapp_init,          // tp_init
                                  0,                                  // tp_alloc
                                  (newfunc) thermapp_new,            // tp_new
                                  0,                                  // tp_free
                                  0,                                  // tp_is_gc
                                  0                                   // tp_bases
    };

// module's functions - there are none
    PyMethodDef moduleMethods[] = {

                                    //our python
                                    {
                                      nullptr } /* Sentinel */
    };

}                                   // anonymous namespace used for local scope declarations

/* Module init function */
PyMODINIT_FUNC initpyipccap(void)
{
    PyObject *module;

    module = Py_InitModule3("pyipccap",
                            moduleMethods,
                            "EYEROP Capture IPC module");

    if (nullptr != module)
    {
        if (PyType_Ready(&thermappType) == EXIT_SUCCESS)
        {
            Py_INCREF(&thermappType);
            PyModule_AddObject(module,
                               "thermapp",
                               (PyObject *) &thermappType);

            PyModule_AddStringConstant(module,
                                       "VERSION",
                                       libop_version());
            PyModule_AddStringConstant(module,
                                       "__version__",
                                       libop_version());
            PyModule_AddStringConstant(module,
                                       "__copyright__",
                                       "Copyright (C) Ogal Optronics Systems Ltd.");
            PyModule_AddStringConstant(module,
                                       "__author__",
                                       "Berger Andrey");
        }                                   //NO NEED FOR ELSE
    }                                   //NO NEED FOR ELSE
}

namespace
{

    /**************************************************************************************/
// private functions implementation
    /**************************************************************************************/
    PyObject * thermapp_new(PyTypeObject *type, PyObject *args, PyObject *kwlist)
    {
        t_thermapp *self = nullptr;

        self = (t_thermapp *) type->tp_alloc(type,
                                             0);
        if (self == nullptr)
        {
            ERRPRINTF("new returned null pointer \n");
        }

        return (PyObject *) self;
    }

    int thermapp_init(t_thermapp *self, PyObject *args, PyObject *keywords)
    {
        DPRINTF("Thermapp_init: initializing\n");
	t_object_name name;
        t_ipccap * ipccap = nullptr;

        //just names for module parameters
        std::string name_str = "name";
        std::string ImageQ_str = "ImageQ";

        int nOutputImageQ;
        static char *keyword_list[] = {
                                        (char *) name_str.c_str(),
                                        (char *) ImageQ_str.c_str(),
                                        NULL };
        DPRINTF("Thremapp_init: name (arg) = %s\n", name.name.c_str());

        //parse input parameters and assign to their variables
        if (PyArg_ParseTupleAndKeywords(args,
                                        keywords,
                                        "O&|i",
                                        keyword_list,
                                        &convert_name_param,
                                        &name,
                                        &nOutputImageQ))
        {
            ipccap = new (t_ipccap);
            if (ipccap != nullptr)
            {
                if (name.isNone)
                {
                    //get default name
                    ipccap->ipcNmae = "/ipc0";
                }
                else
                {
                    // (name != None) ==> use name supplied by the caller.
                    //  It was already converted to std string by convert_name_param().
                    DPRINTF("name.name is %s\n", name.name.c_str());
                    ipccap->ipcNmae = name.name;
                }
                DPRINTF("Thermapp_init: ipccap->ipcNmae.c_str() = %s\n", ipccap->ipcNmae.c_str());
                ipccap->nOutputImageQ = nOutputImageQ;

                self->ipcCapHandle = (uintmax_t) ipccap;
                return 0;
            }
        }
        DPRINTF("Thermapp_init: Parse FAILED, return -1\n");
        return -1;

    }

    void thermapp_dealloc(t_thermapp *self)
    {
        Py_TYPE(self)->tp_free((PyObject*) self);

    }

    PyObject * pyipccap_get_data(t_thermapp *self, PyObject *args)
    {
        t_ipccap * ipccap = (t_ipccap *) self->ipcCapHandle;

        uint8_t* pImage = static_cast<uint8_t*>(ipccap->client.lock_next_image_for_read());
        if (nullptr != pImage)
        {
            uint16_t *imageHeader = (uint16_t*) pImage;
            //  pImage += 64; // possible to Skip header

            self->serialNumber = imageHeader[5] | (uint32_t(imageHeader[6]) << 16);

            self->imageHeight = imageHeader[11];
            self->imageWidth = imageHeader[12];
            self->imageId = imageHeader[26] | (uint32_t(imageHeader[27]) << 16);

            DPRINTF("Got our frame %u with dimensions (%d,%d)\n",
                    self->imageId,
                    self->imageWidth,
                    self->imageHeight);
            // Make sure we release the locked image

            PyObject * py_pImage = PyByteArray_FromStringAndSize((const char *) pImage,
                                                                 (Py_ssize_t) (64
                                                                               + (self->imageHeight
                                                                                  * self->imageWidth
                                                                                  * 2)));
            ipccap->client.release_locked_image();
            if (nullptr != py_pImage)
            {
                return py_pImage;
            }
        } // NO NEED FOR ELSE
        Py_RETURN_NONE;

    }

    PyObject * pyipccap_open_client(t_thermapp *self, PyObject *args)
    {
        DPRINTF("Open_client: initialize\n");
	
	t_ipccap * ipccap = (t_ipccap *) self->ipcCapHandle;
        DPRINTF("Open_client: ipccap->ipcNmae.c_str() = %s\n", ipccap->ipcNmae.c_str());
        DPRINTF("Open_client: ipccap->nOutputImageQ = %d\n", ipccap->nOutputImageQ);
        DPRINTF("Open_client: ipccap->memorySize = %u\n", memorySize);
        
        if (ipccap->client.initialize(ipccap->ipcNmae,
                                      ipccap->nOutputImageQ,
                                      memorySize))
        {
            DPRINTF("Open_client: client.initialize SUCCEEDED\n");
            ipccap->client.sync(); // Make sure we start at a known time
            return Py_BuildValue("i",
                                 0);
        }
        else
        {
            DPRINTF("Open_client: client.initialize FAILED\n");
            return Py_BuildValue("i",
                                 -1);
        }

    }

    int convert_name_param(PyObject *pyNameParam, void *checkedName)
    {

        int rc = 0;
        t_object_name *p_name = (t_object_name *) checkedName;

        DPRINTF("PyBytes_Check() = %d \n", PyBytes_Check(pyNameParam));
        DPRINTF("PyString_Check() = %d \n", PyString_Check(pyNameParam));
        DPRINTF("PyUnicode_Check() = %d \n", PyUnicode_Check(pyNameParam));

        p_name->isNone = false;

        // The name can be None or a Python string
        if (Py_None == pyNameParam)
        {
            DPRINTF("name is None\n");
            //the case of no name parameter will take action later in code
            rc = 1;
            p_name->isNone = true;

        }
        else if (PyString_Check(pyNameParam) || PyUnicode_Check(pyNameParam))
        {
            DPRINTF("name is string or unicode\n");

            p_name->name = PyString_AsString(pyNameParam);
            DPRINTF("p_name is %s\n", p_name->name.c_str());
        }
        else
            PyErr_SetString(PyExc_TypeError,
                            "Name must be None or a string");
        return rc;
    }

}            // anonymous namespace used for local scope declarations
