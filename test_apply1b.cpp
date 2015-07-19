#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

static const char *kernelSource = R"DELIM(
  kernel void test(int offset, int totalN, global float*_out) {
    int linearId = get_global_id(0) + offset;
    if(linearId < totalN) {
      float out = _out[linearId];
      _out[linearId] = {{operation}};
    }
  }
)DELIM";

void test(EasyCL *cl, int numLaunches, int vectorSize, string operation = "+") {
  string arrayType = "float";
  if(vectorSize > 1) {
    arrayType += easycl::toString(vectorSize);
  }
  int totalN = 128 * 1024 * 1024;
//  int numLaunches = 1;
  int N = totalN / numLaunches;
  string templatedSource = easycl::replaceGlobal(kernelSource, "float", arrayType);
  templatedSource = easycl::replace(templatedSource, "{{operation}}", operation);
//  cout << templatedSource << endl;
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
  double end = StatefulTimer::instance()->getSystemMilliseconds();
  wrapper->copyToHost();
//  cout << "in[10]" << in[10] << endl;
//  cout << "Time, " << numLaunches << " launches: " << (end - start) << "ms" << endl;
  cout << "launches " << numLaunches << " N per launch " << N << " vectorsize=" << vectorSize << " op=" << operation << " time=" << (end - start) << "ms" << endl;

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

void testVectorSize(EasyCL *cl) {
  for( int p = 0; p < 16; p += 8 ) {
    int numLaunches = 1 << p;
    test(cl, numLaunches, 1);
    test(cl, numLaunches, 4);
  }
}

void testOperations(EasyCL *cl) {
  test(cl, 256, 4, "out + 3.3f");
  test(cl, 256, 4, "out * 3.3f");
  test(cl, 256, 4, "out / 3.3f");
  test(cl, 256, 1, "out + 3.3f");
  test(cl, 256, 1, "out * 3.3f");
  test(cl, 256, 1, "out / 3.3f");
  test(cl, 256, 1, "native_exp(out)");
  test(cl, 256, 1, "tanh(out)");
//  test(cl, 256, 1, "%");
}

int main(int argc, char *argv[]) {
  int gpu = 0;
  if( argc == 2 ) {
    gpu = atoi(argv[1]);
  }
  cout << "using gpu " << gpu << endl;
  EasyCL *cl = EasyCL::createForIndexedGpu(gpu);
//  testVectorSize(cl);
  testOperations(cl);
  delete cl;
  return 0;
}


