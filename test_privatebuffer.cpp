#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

static const char *kernelSource = R"DELIM(
  kernel void test(int totalN, global float*out) {
    int linearId = get_global_id(0) * {{privatesize}};
    if(linearId + {{privatesize}} < totalN) {
      float _buffer[{{privatesize}}];
      #pragma unroll
      for( int i = 0; i < {{privatesize}}; i++ ) {
        _buffer[i] = out[linearId + i];
      }
      for( int i = 0; i < {{privatesize}}; i++ ) {
        out[linearId +i] = _buffer[i] + 3.3f;
      }
    }
  }
)DELIM";

void test(EasyCL *cl, int privateSize) {
  int totalN = 128 * 1024 * 1024;
  string templatedSource = easycl::replaceGlobal(kernelSource, "{{privatesize}}", easycl::toString(privateSize));
  CLKernel *kernel = cl->buildKernelFromString(templatedSource, "test", "");
  int workgroupSize = 64;
  int numWorkgroups = (totalN / privateSize + workgroupSize - 1) / workgroupSize;

  float *in = new float[totalN];
  float *inOut = new float[totalN];
  for( int i = 0; i < totalN; i++ ) {
      in[i] = inOut[i] = (i + 4) % 1000000;
  }
  CLWrapper *wrapper = cl->wrap(totalN, inOut);
  wrapper->copyToDevice();

  cl->finish();

  double start = StatefulTimer::instance()->getSystemMilliseconds();
  kernel->in(totalN);
  kernel->inout(wrapper);
  kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
  cl->finish();
  double end = StatefulTimer::instance()->getSystemMilliseconds();
  wrapper->copyToHost();
  cout << "privateSize=" << privateSize << " time=" << (end - start) << "ms" << endl;

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

int main(int argc, char *argv[]) {
  int gpu = 0;
  if( argc == 2 ) {
    gpu = atoi(argv[1]);
  }
  cout << "using gpu " << gpu << endl;
  EasyCL *cl = EasyCL::createForIndexedGpu(gpu);
  cl->setProfiling(true);
  for(int p = 0; p < 16; p++) {
    test(cl, 1<<p);
    cl->dumpProfiling();
  }
  cl->dumpProfiling();
  delete cl;
  return 0;
}


