#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

static const char *kernelSource = R"DELIM(
  kernel void test(int offset, int totalN, global float*out) {
    int linearId = get_global_id(0) + offset;
    if(linearId < totalN) {
      out[linearId] = out[linearId] + 3.3f;
    }
  }
)DELIM";

void test(EasyCL *cl, int workgroupSize) {
  int numLaunches = 256;
  int vectorSize = 4;
  string arrayType = "float";
  if(vectorSize > 1) {
    arrayType += easycl::toString(vectorSize);
  }
  int totalN = 128 * 1024 * 1024;
  int N = totalN / numLaunches;
  string templatedSource = easycl::replace(kernelSource, "float", arrayType);
  CLKernel *kernel = cl->buildKernelFromString(templatedSource, "test", "");
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
  cout << "launches " << numLaunches << " N per launch " << N << " workgroupSize=" << workgroupSize << " time=" << (end - start) << "ms" << endl;

  int errorCount = 0;
  for( int i = 0; i < totalN; i++ ) {
    if(inOut[i] != in[i] + 3.3f ) {
      errorCount++;
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

void testOperations(EasyCL *cl) {
  test(cl, 64);
  cl->dumpProfiling();
  test(cl, 128);
  cl->dumpProfiling();
  test(cl, 256);
  cl->dumpProfiling();
}

int main(int argc, char *argv[]) {
  int gpu = 0;
  if( argc == 2 ) {
    gpu = atoi(argv[1]);
  }
  cout << "using gpu " << gpu << endl;
  EasyCL *cl = EasyCL::createForIndexedGpu(gpu);
  cl->setProfiling(true);
  testOperations(cl);
  cl->dumpProfiling();
  delete cl;
  return 0;
}


