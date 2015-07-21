#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

static const char *kernelSource = R"DELIM(
  kernel void test(int totalN,
      int out_offset,
      int out_dims,
      int out_dim0,
      int out_dim1,
      int out_stride0,
      int out_stride1,
      global float*out_data,
      int in1_offset,
      int in1_dims,
      int in1_dim0,
      int in1_dim1,
      int in1_stride0,
      int in1_stride1,
      global float *in1_data,
      int in2_offset,
      int in2_dims,
      int in2_dim0,
      int in2_dim1,
      int in2_stride0,
      int in2_stride1,
      global float *in2_data
      ) {
    int linearId = get_global_id(0);
    if(linearId < totalN) {
      out_data[linearId] = in1_data[linearId] * in2_data[linearId];
    }
  }
)DELIM";

void test(EasyCL *cl, int its, int size, bool reuseStructBuffers) {
  int totalN = size;
  string templatedSource = kernelSource;
  CLKernel *kernel = cl->buildKernelFromString(templatedSource, "test", "");
  const int workgroupSize = 64;
  int numWorkgroups = (totalN + workgroupSize - 1) / workgroupSize;

  float *out = new float[totalN];
  float *in1 = new float[totalN];
  float *in2 = new float[totalN];
  for( int i = 0; i < totalN; i++ ) {
      in1[i] = (i + 4) % 1000000;
      in2[i] = (i + 6) % 1000000;
  }
  CLWrapper *outwrap = cl->wrap(totalN, out);
  CLWrapper *in1wrap = cl->wrap(totalN, in1);
  CLWrapper *in2wrap = cl->wrap(totalN, in2);
  in1wrap->copyToDevice();
  in2wrap->copyToDevice();
  outwrap->createOnDevice();

  cl->finish();
  cl->dumpProfiling();

  double start = StatefulTimer::instance()->getSystemMilliseconds();
  for(int it = 0; it < its; it++) {
    kernel->in(totalN);

    kernel->in(0);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->out(outwrap);
    kernel->in(0);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(in1wrap);
    kernel->in(0);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(2);
    kernel->in(in2wrap);

    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
  }
  cl->finish();
  double end = StatefulTimer::instance()->getSystemMilliseconds();
  cl->dumpProfiling();
  cout << "its=" << its << " size=" << size << " reusestructbuffers=" << reuseStructBuffers << " time=" << (end - start) << "ms" << endl;
  outwrap->copyToHost();
  cl->finish();
  int errorCount = 0;
  for( int i = 0; i < totalN; i++ ) {
    float targetValue = in1[i] * in2[i];
    if(abs(out[i] - targetValue)> 0.1f) {
      errorCount++;
      if( errorCount < 20 ) {
        cout << "out[" << i << "]" << " != " << targetValue << endl;
        cout << abs(out[i] - targetValue) << endl;
      }
    }
  }
//  cout << endl;
  if( errorCount > 0 ) {
    cout << "errors: " << errorCount << " out of totalN=" << totalN << endl;
  } else {
  }

  delete outwrap;
  delete in1wrap;
  delete in2wrap;
  delete[] in1;
  delete[] in2;
  delete[] out;
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
  test(cl, 900, 6400, true);
  test(cl, 9000, 6400, true);
  cl->dumpProfiling();
  delete cl;
  return 0;
}


