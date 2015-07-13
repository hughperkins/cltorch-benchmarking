#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

// let's imagine we have a 32x32 transposed matrix
// now of course, we could still process in memory order, but on apply2,
// with two tensors, one transposed, one not, thats not the case
// let's measure what happens if we use column-major order on a tensor
// stored as row-major (of course, this depends on the exact tensor
// dimensions too now, but have to start somewhere...)
//
// nuance: since we have a huge block of memory we're just going to image it's
// one huge 2d tensor, either in rows of eg 32, or when transposed, in
// columns of 32, actually of {{stride}}
// so, if stride is 1:
// row = linearId / 
//
// untransposed, eg R x 4:
// stride: {4, 1}
// size: {R, 4}
// 0 1 2 3
// 4 5 6 7
// 8 9 10 11
// linearid
// row = linearid / 4 = linearid / stride0 = linearid / size1
// col = linearid % 4 = linearid % stride0 = linearid % size1
// offset = row * stride0 + col = (linearid / stride0) * stride0 + linearid % stride0
// not sure how to simplify this using standard maths rules, but seems clear that: 
//       (i / a) * a + i % a == i , for natural i, a
//
// transposed, eg 4 x R:
// stride: {1, 4}
// size: {4, R}
// 0 4 8 12 16 20   .. 4*R-4
// 1 5 9 13 17 21   .. 4*R-3
// 2 6 10 14 18 22  .. 4*R-2
// 3 7 11 15 19 23  .. 4*R-1
// linearid
// row = linearid % 4 = linearid % stride1 = linearid % size0
// col = linearid / 4 = linearid / stride1 = linearid / size0
// offset = row * stride0 + col * stride1
//         = ( linearid % size0 )
//
// 3 dimensions
// size = (s0, s1, s2)
// stride = (r0, r1, r2) = (s2*s1, s2, 1)
// coord = (x0, x1, x2)
// storageoffset = x0*r0 + x1*r1 + x2*r2 = x0*s1*s2 + x1*s2 + x2
// after transpose, (x0,x1,x2)=>(x1,x0,x2) storageoffset = x0*r1 + x1*r0 + x2*r2 = x1*s1*s2 + x0*s2 + x2
// except, sizes transposed too, (s0,s1,s2)=>(s1,s0,s2), so => x1*s0*s2 + x0*s2 + x2
// 
// or transpose (x0,x1,x2)=>(x2,x1,x0) (s0,s1,s2)=>(s2,s1,s0),  so =>x2*s1*s0 + x1*s0 + x0
//
// given storageoffset, and strides (r0,r1,r2), and sizes (s0,s1,s2), and we want
// to know coords (x0,x1,x2)
// coords, given storageoffset p:
//   x2 = p % s2
//   x1 = (p/s2) % s1
//   x0 = (p/s2/s1)
//
// in 2d:
// x1 = p % s1
// x0 = p / s1
//
// let's imagine our storageoffset is based on (s0, s1) (r0, r1), but we want
// to access in transposed order ie, based on (s1, s0)(r1, r0)
static const char *kernelSource = R"DELIM(
  #define STRIDE0 {{stride0}}
  #define STRIDE1 {{stride1}}
  #define SIZE1 {{size1}}
  kernel void test(int offset, int totalN, global float*_out) {
    int linearId = get_global_id(0) + offset;
    if(linearId < totalN) {
      int x1 = linearId % SIZE1;
      int x0 = linearId / SIZE1;
      int storageOffset = 0;
      if( {{transposed}} ) {
        storageOffset = x0 * STRIDE1 + x1 * STRIDE0;
      } else {
        storageOffset = x0 * STRIDE0 + x1 * STRIDE1;
      }
      float out = _out[storageOffset];
      _out[storageOffset] = {{operation}};
    }
  }
)DELIM";

string boolToString(bool value) {
  if(value) {
    return "true";
  }
  return "false";
}

void test(EasyCL *cl, int numLaunches, int vectorSize, string operation = "+", bool transposed=false) {
  string arrayType = "float";
  if(vectorSize > 1) {
    arrayType += easycl::toString(vectorSize);
  }
  int totalN = 128 * 1024 * 1024;
//  int numLaunches = 1;
  int N = totalN / numLaunches;

  int size0 = N / 32;
  int size1 = 32;

  int stride0 = 32;
  int stride1 = 1;

  string templatedSource = easycl::replaceGlobal(kernelSource, "float", arrayType);
  templatedSource = easycl::replace(templatedSource, "{{operation}}", operation);
  templatedSource = easycl::replace(templatedSource, "{{stride0}}", easycl::toString(stride0));
  templatedSource = easycl::replace(templatedSource, "{{stride1}}", easycl::toString(stride1));
  templatedSource = easycl::replace(templatedSource, "{{size1}}", easycl::toString(size1));
  templatedSource = easycl::replace(templatedSource, "{{transposed}}", boolToString(transposed));
  cout << templatedSource << endl;
  CLKernel *kernel = cl->buildKernelFromString(templatedSource, "test", "");
  const int workgroupSize = 64;
  int numWorkgroups = (N / vectorSize + workgroupSize - 1) / workgroupSize;

  float *in = new float[totalN];
  float *inOut = new float[totalN];
  for( int i = 0; i < totalN; i++ ) {
      in[i] = inOut[i] = (i + 4) % 1000000;
  }
  CLWrapper *wrapper = cl->wrap(totalN, inOut);
  wrapper->copyToDevice();

  cl->finish();

  double start = StatefulTimer::instance()->getSystemMilliseconds();
  for( int i = 0; i < numLaunches; i++ ) {
    kernel->in(N * i / vectorSize);
    kernel->in(totalN / vectorSize);
    kernel->inout(wrapper);
    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
  }
  cl->finish();
  wrapper->copyToHost();
//  cout << "in[10]" << in[10] << endl;
  double end = StatefulTimer::instance()->getSystemMilliseconds();
//  cout << "Time, " << numLaunches << " launches: " << (end - start) << "ms" << endl;
  cout << "launches " << numLaunches << " N per launch " << N << " vectorsize=" << vectorSize << " op=" << operation << " t=" << transposed << " time=" << (end - start) << "ms" << endl;

  int errorCount = 0;
  if( operation == "out + 3.3f" ) {
    for( int i = 0; i < totalN; i++ ) {
      if(inOut[i] != in[i] + 3.3f ) {
        errorCount++;
  //      if( errorCount < 20 ) {
  //        cout << in[i] << " != " << (float)(i+4+1) << endl;
  //      }
      }
    }
  }
//  cout << endl;
  if( errorCount > 0 ) {
    cout << "errors: " << errorCount << " out of totalN=" << totalN << endl;
  } else {
//    cout << "No errors detected" << endl;
  }

  delete wrapper;
  delete[] in;
  delete[] inOut;
  delete kernel;
}

void testTranspose(EasyCL *cl) {
  test(cl, 256, 4, "out + 3.3f", false);
  test(cl, 256, 4, "out + 3.3f", true);
}

int main(int argc, char *argv[]) {
  int gpu = 0;
  if( argc == 2 ) {
    gpu = atoi(argv[1]);
  }
  cout << "using gpu " << gpu << endl;
  EasyCL *cl = EasyCL::createForIndexedGpu(gpu);
//  testVectorSize(cl);
  testTranspose(cl);
  delete cl;
  return 0;
}


