#include "FilterTIG.hpp"
#include <cstdint>
#include <iostream>
#include <assert.h>
#include "numpy/ndarrayobject.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class ScopedGILRelease
{
public:
    inline ScopedGILRelease()
    {
      //m_thread_state = PyEval_SaveThread();
    }

    inline ~ScopedGILRelease()
    {
      //PyEval_RestoreThread(m_thread_state);
      //m_thread_state = NULL;
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

FilterTIG::FilterTIG(){
	import_array();
}

void FilterTIG::update(PyObject* weights_arr){

	//Release GIL
	ScopedGILRelease scoped;

	Mat w1f;
	pyopencv_to(weights_arr, w1f);

	CV_Assert(w1f.cols * w1f.rows == D && w1f.type() == CV_32F && w1f.isContinuous());
	float b[D], residuals[D];
	memcpy(residuals, w1f.data, sizeof(float)*D);
	for (int i = 0; i < NUM_COMP; i++){
		float avg = 0;
		for (int j = 0; j < D; j++){
			b[j] = residuals[j] >= 0.0f ? 1.0f : -1.0f;
			avg += residuals[j] * b[j];
		}
		avg /= D;
		_coeffs1[i] = avg, _coeffs2[i] = avg*2, _coeffs4[i] = avg*4, _coeffs8[i] = avg*8;
		for (int j = 0; j < D; j++)
			residuals[j] -= avg*b[j];
		UINT64 tig = 0;
		for (int j = 0; j < D; j++)
			tig = (tig << 1) | (b[j] > 0 ? 1 : 0);
		_bTIGs[i] = tig;
	}
}

void FilterTIG::reconstruct(PyObject* weights_arr){

	//Release GIL
	ScopedGILRelease scoped;

	Mat w1f;
	pyopencv_to(weights_arr, w1f);

	w1f = Mat::zeros(8, 8, CV_32F);
	float *weight = (float*)w1f.data;
	for (int i = 0; i < NUM_COMP; i++){
		UINT64 tig = _bTIGs[i];
		for (int j = 0; j < D; j++)
			weight[j] += _coeffs1[i] * (((tig >> (63-j)) & 1) ? 1 : -1);
	}
}

// For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
// Please refer to my paper for definition of the variables used in this function
PyObject* FilterTIG::matchTemplate(PyObject* grad){

	//Release GIL
	ScopedGILRelease scoped;

	Mat mag1u;

	pyopencv_to(grad, mag1u);

	const int H = mag1u.rows, W = mag1u.cols;
	const Size sz(W+1, H+1); // Expand original size to avoid dealing with boundary conditions
	Mat_<INT64> Tig1 = Mat_<INT64>::zeros(sz), Tig2 = Mat_<INT64>::zeros(sz);
	Mat_<INT64> Tig4 = Mat_<INT64>::zeros(sz), Tig8 = Mat_<INT64>::zeros(sz);
	Mat_<byte> Row1 = Mat_<byte>::zeros(sz), Row2 = Mat_<byte>::zeros(sz);
	Mat_<byte> Row4 = Mat_<byte>::zeros(sz), Row8 = Mat_<byte>::zeros(sz);
	Mat_<float> scores(sz);
	for(int y = 1; y <= H; y++){ 
		const byte* G = mag1u.ptr<byte>(y-1);
		INT64* T1 = Tig1.ptr<INT64>(y); // Binary TIG of current row
		INT64* T2 = Tig2.ptr<INT64>(y);
		INT64* T4 = Tig4.ptr<INT64>(y);
		INT64* T8 = Tig8.ptr<INT64>(y);
		INT64* Tu1 = Tig1.ptr<INT64>(y-1); // Binary TIG of upper row
		INT64* Tu2 = Tig2.ptr<INT64>(y-1);
		INT64* Tu4 = Tig4.ptr<INT64>(y-1);
		INT64* Tu8 = Tig8.ptr<INT64>(y-1);
		byte* R1 = Row1.ptr<byte>(y);
		byte* R2 = Row2.ptr<byte>(y);
		byte* R4 = Row4.ptr<byte>(y);
		byte* R8 = Row8.ptr<byte>(y);
		float *s = scores.ptr<float>(y);
		for (int x = 1; x <= W; x++) {
			byte g = G[x-1];
			R1[x] = (R1[x-1] << 1) | ((g >> 4) & 1);
			R2[x] = (R2[x-1] << 1) | ((g >> 5) & 1);
			R4[x] = (R4[x-1] << 1) | ((g >> 6) & 1);
			R8[x] = (R8[x-1] << 1) | ((g >> 7) & 1);
			T1[x] = (Tu1[x] << 8) | R1[x];
			T2[x] = (Tu2[x] << 8) | R2[x];
			T4[x] = (Tu4[x] << 8) | R4[x];
			T8[x] = (Tu8[x] << 8) | R8[x];
			s[x] = dot(T1[x], T2[x], T4[x], T8[x]);
		}
	}
	Mat matchCost1f;
	scores(Rect(8, 8, W-7, H-7)).copyTo(matchCost1f);
	return pyopencv_from(matchCost1f);
}

struct pixel_item {
  float score;
  Point point;
} ;

bool pixel_item_cmp (pixel_item* i,pixel_item* j) { return (i->score>j->score); }

PyObject* FilterTIG::nonMaxSup(PyObject* match_map, int NSS, int maxPoint, bool fast)
{

	pixel_item* item;
	vector<pixel_item*> valPnt;
	vector<pixel_item*> matchCost;

	Mat matchCost1f;
	pyopencv_to(match_map, matchCost1f);

	const int _h = matchCost1f.rows, _w = matchCost1f.cols;
	Mat isMax1u = Mat::ones(_h, _w, CV_8U), costSmooth1f;
	//ValStructVec<float, Point> valPnt;
	//matchCost.reserve(_h * _w);
	//valPnt.reserve(_h * _w);
	if (fast){
		blur(matchCost1f, costSmooth1f, Size(3, 3));
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			const float* ds = costSmooth1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				if (d[c] >= ds[c]){
					//valPnt.pushBack(d[c], Point(c, r));
					item = new pixel_item;
					if(!item)
						return Py_None;
					item->score = d[c];
					item->point.x = c;
					item->point.y = r;
					valPnt.push_back(item);
				}
		}
	}
	else{
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			for (int c = 0; c < _w; c++){
				//valPnt.pushBack(d[c], Point(c, r));
				item = new pixel_item;
				if(!item)
					return Py_None;
				item->score = d[c];
				item->point.x = c;
				item->point.y = r;
				valPnt.push_back(item);
			}
		}
	}

	sort( valPnt.begin(), valPnt.end(), pixel_item_cmp);
	for (int i = 0; i < valPnt.size(); i++){
		Point pnt = valPnt[i]->point;
		if (isMax1u.at<byte>(pnt)){
			item = new pixel_item;
			if(!item)
				return Py_None;
			item->score = valPnt[i]->score;
			item->point.x = pnt.x;
			item->point.y = pnt.y;
			matchCost.push_back(item);
			for (int dy = -NSS; dy <= NSS; dy++) for (int dx = -NSS; dx <= NSS; dx++){
				Point neighbor = pnt + Point(dx, dy);
				if (!CHK_IND(neighbor))
					continue;
				isMax1u.at<byte>(neighbor) = false;
			}
		}
		if (matchCost.size() >= maxPoint)
			break;
	}

	PyObject *pyMatchCost = PyList_New(0);
	if(!pyMatchCost)
		return Py_None;

	PyObject* py_point;
	PyObject* py_item;
	for (int i = 0; i < matchCost.size(); i++){
		py_point = Py_BuildValue("(ii)", matchCost[i]->point.x, matchCost[i]->point.y);
		py_item = Py_BuildValue("(Of)", py_point, matchCost[i]->score);
		PyList_Append(pyMatchCost, py_item);
	}

	return pyMatchCost;
}
