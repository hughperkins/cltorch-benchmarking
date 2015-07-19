#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "util/StatefulTimer.h"

static const char *kernelSource = R"DELIM(
  kernel void test(int offset, int totalN, global float*out) {
    int linearId = get_global_id(0) + offset;
    if(linearId < totalN) {
      out[linearId] = out[linearId] + 1.0f;
    }
  }
)DELIM";

void test(EasyCL *cl, int totalN, int numLaunches) {
//  int numLaunches = 1;
  int N = totalN / numLaunches;
//  cout << kernelSource << endl;
  CLKernel *kernel = cl->buildKernelFromString(kernelSource, "test", "");
  const int workgroupSize = 64;
  int numWorkgroups = (N + workgroupSize - 1) / workgroupSize;

  float *in = new float[totalN];
  for( int i = 0; i < totalN; i++ ) {
      in[i] = i + 4;
  }
  CLWrapper *wrapper = cl->wrap(totalN, in);
  wrapper->copyToDevice();

  cl->finish();
  cl->dumpProfiling();

  for(int it = 0; it < 1; it++ ) {
    double start = StatefulTimer::instance()->getSystemMilliseconds();
    for( int i = 0; i < numLaunches; i++ ) {
      kernel->in(N * i);
      kernel->in(totalN);
      kernel->inout(wrapper);
      kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
    }
    cl->finish();
    cl->dumpProfiling();
    wrapper->copyToHost();
  //  cout << "in[10]" << in[10] << endl;
    double end = StatefulTimer::instance()->getSystemMilliseconds();
  //  cout << "Time, " << numLaunches << " launches: " << (end - start) << "ms" << endl;
    cout << "it=" << it << " totalN " << totalN << " launches " << numLaunches << " N per launch " << N << " time=" << (end - start) << "ms" << endl;
  }

  delete wrapper;
  delete[] in;
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
  for( int p = 0; p <= 14; p += 2 ) {
    for(int totalN = 32 * 1024 * 1024; totalN <= 256 * 1024 * 1024; totalN *= 2 ) {
      int numLaunches = 1 << p;
      test(cl, totalN, numLaunches);
    }
  }
  delete cl;
  return 0;
}


