#include "py_liblinear.hpp"

#include <cstdint>
#include <assert.h>
#include <string>
#include <iostream>
#include "numpy/ndarrayobject.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "LibLinear/linear.h"

using namespace cv;
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class ScopedGILRelease
{
public:
    inline ScopedGILRelease()
    {
        m_thread_state = PyEval_SaveThread();
    }

    inline ~ScopedGILRelease()
    {
        PyEval_RestoreThread(m_thread_state);
       m_thread_state = NULL;
    }

private:
    PyThreadState * m_thread_state;
};

// The following conversion functions are taken from OpenCV's cv2.cpp file inside modules/python/src2 folder.
static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

struct ArgInfo
{
    const char * name;
    bool outputarg;
    // more fields may be added if necessary

    ArgInfo(const char * name_, bool outputarg_)
        : name(name_)
        , outputarg(outputarg_) {}

    // to match with older pyopencv_to function signature
    operator const char *() const { return name; }
};

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

using namespace cv;

typedef std::vector<uchar> vector_uchar;
typedef std::vector<char> vector_char;
typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;
typedef std::vector<double> vector_double;
typedef std::vector<Point> vector_Point;
typedef std::vector<Point2f> vector_Point2f;
typedef std::vector<Vec2f> vector_Vec2f;
typedef std::vector<Vec3f> vector_Vec3f;
typedef std::vector<Vec4f> vector_Vec4f;
typedef std::vector<Vec6f> vector_Vec6f;
typedef std::vector<Vec4i> vector_Vec4i;
typedef std::vector<Rect> vector_Rect;
typedef std::vector<KeyPoint> vector_KeyPoint;
typedef std::vector<Mat> vector_Mat;
typedef std::vector<DMatch> vector_DMatch;
typedef std::vector<String> vector_String;
typedef std::vector<Scalar> vector_Scalar;

typedef std::vector<std::vector<char> > vector_vector_char;
typedef std::vector<std::vector<Point> > vector_vector_Point;
typedef std::vector<std::vector<Point2f> > vector_vector_Point2f;
typedef std::vector<std::vector<Point3f> > vector_vector_Point3f;
typedef std::vector<std::vector<DMatch> > vector_vector_DMatch;

#ifdef HAVE_OPENCV_FEATURES2D
typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;
#endif

#ifdef HAVE_OPENCV_FLANN
typedef cvflann::flann_distance_t cvflann_flann_distance_t;
typedef cvflann::flann_algorithm_t cvflann_flann_algorithm_t;
#endif

#ifdef HAVE_OPENCV_STITCHING
typedef Stitcher::Status Status;
#endif

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() { stdAllocator = Mat::getStdAllocator(); }
    ~NumpyAllocator() {}

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
    {
        UMatData* u = new UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( int i = 0; i < dims - 1; i++ )
            step[i] = (size_t)_strides[i];
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const
    {
        if( data != 0 )
        {
            CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
            _sizes[dims++] = cn;
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        return allocate(o, dims0, sizes, type, step);
    }

    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const
    {
        if(u)
        {
            PyEnsureGIL gil;
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};

NumpyAllocator g_numpyAllocator;


template<typename T> static
bool pyopencv_to(PyObject* obj, T& p, const char* name = "<unknown>");

template<typename T> static
PyObject* pyopencv_from(const T& src);

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

// special case, when the convertor needs full ArgInfo structure
static bool pyopencv_to(PyObject* o, Mat& m, const ArgInfo info)
{
    bool allowND = true;
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if( PyInt_Check(o) )
    {
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyFloat_Check(o) )
    {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyTuple_Check(o) )
    {
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = Mat(sz, 1, CV_64F);
        for( i = 0; i < sz; i++ )
        {
            PyObject* oi = PyTuple_GET_ITEM(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else
            {
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || type == NPY_LONG )
        {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        }
        else
        {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for( int i = ndims-1; i >= 0 && !needcopy; i-- )
    {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        if( (i == ndims-1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    if( ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2] )
        needcopy = true;

    if (needcopy)
    {
        if (info.outputarg)
        {
            failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
            return false;
        }

        if( needcast ) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    // handle degenerate case
    if( ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ismultichannel )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    m.addref();

    if( !needcopy )
    {
        Py_INCREF(o);
    }
    m.allocator = &g_numpyAllocator;

    return true;
}

template<>
bool pyopencv_to(PyObject* o, Mat& m, const char* name)
{
    return pyopencv_to(o, m, ArgInfo(name, 0));
}

template<>
PyObject* pyopencv_from(const Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->u || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}

template<>
bool pyopencv_to(PyObject *o, Scalar& s, const char *name)
{
    if(!o || o == Py_None)
        return true;
    if (PySequence_Check(o)) {
        PyObject *fi = PySequence_Fast(o, name);
        if (fi == NULL)
            return false;
        if (4 < PySequence_Fast_GET_SIZE(fi))
        {
            failmsg("Scalar value for argument '%s' is longer than 4", name);
            return false;
        }
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
            if (PyFloat_Check(item) || PyInt_Check(item)) {
                s[(int)i] = PyFloat_AsDouble(item);
            } else {
                failmsg("Scalar value for argument '%s' is not numeric", name);
                return false;
            }
        }
        Py_DECREF(fi);
    } else {
        if (PyFloat_Check(o) || PyInt_Check(o)) {
            s[0] = PyFloat_AsDouble(o);
        } else {
            failmsg("Scalar value for argument '%s' is not numeric", name);
            return false;
        }
    }
    return true;
}

template<>
PyObject* pyopencv_from(const Scalar& src)
{
    return Py_BuildValue("(dddd)", src[0], src[1], src[2], src[3]);
}

template<>
PyObject* pyopencv_from(const bool& value)
{
    return PyBool_FromLong(value);
}

PyLibLinear::PyLibLinear(){
	import_array(); // This is a function from NumPy that MUST be called.
}

PyObject* PyLibLinear::matRead(std::string filename){

	//Release GIL
	ScopedGILRelease scoped;

	FILE* f = fopen(filename.c_str(), "rb");
	if (f == NULL)
		return Py_None;
	char buf[8];
	int pre = fread(buf,sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		printf("Invalidate CvMat data file %s\n", filename.c_str());
		return Py_None;
	}
	int headData[3]; // Width, height, type
	fread(headData, sizeof(int), 3, f);
	Mat M(headData[1], headData[0], headData[2]);
	fread(M.data, sizeof(char), M.step * M.rows, f);
	fclose(f);

	return pyopencv_from(M);
}

void PyLibLinear::matWrite(std::string filename, PyObject* arr){
	return;
}

void print_null(const char *s){}

PyObject* PyLibLinear::trainSVM(PyObject* feat_arr, PyObject* Y, int sT, double C, double bias, double eps)
{

	//Release GIL
	ScopedGILRelease scoped;

	Mat X1f;
	pyopencv_to(feat_arr,X1f);

	// Set SVM parameters
	parameter param; {
		param.solver_type = sT; // L2R_L2LOSS_SVC_DUAL;
		param.C = C;
		param.eps = eps; // see setting below
		//param.p = 0.1;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		set_print_string_function(print_null);
		CV_Assert(PyArray_NDIM((PyArrayObject *)Y)==1);
		CV_Assert(X1f.rows == PyArray_SHAPE((PyArrayObject *)Y)[0]);
		CV_Assert(X1f.type() == CV_32F);
	}

	// Initialize a problem
	feature_node *x_space = NULL;
	problem prob;{
		prob.l = X1f.rows;
		prob.bias = bias;
		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(feature_node*, prob.l);
		const int DIM_FEA = X1f.cols;
		prob.n = DIM_FEA + (bias >= 0 ? 1 : 0);
		x_space = Malloc(feature_node, (prob.n + 1) * prob.l);
		int j = 0;
		for (int i = 0; i < prob.l; i++){
			prob.y[i] = (double) *((float*) PyArray_GETPTR1((PyArrayObject *)Y,i));
			prob.x[i] = &x_space[j];
			const float* xData = X1f.ptr<float>(i);
			for (int k = 0; k < DIM_FEA; k++){
				x_space[j].index = k + 1;
				x_space[j++].value = xData[k];
			}
			if (bias >= 0){
				x_space[j].index = prob.n;
				x_space[j++].value = bias;
			}
			x_space[j++].index = -1;
		}
		CV_Assert(j == (prob.n + 1) * prob.l);
	}

	// Training SVM for current problem
	const char*  error_msg = check_parameter(&prob, &param);
	if(error_msg){
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	model *svmModel = train(&prob, &param);
	Mat wMat(1, prob.n, CV_64F, svmModel->w);
	wMat.convertTo(wMat, CV_32F);
	free_and_destroy_model(&svmModel);
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	return pyopencv_from(wMat);
}
