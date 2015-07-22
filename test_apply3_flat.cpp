#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"
#include "templates/TemplatedKernel.h"

static const char *kernelSource = R"DELIM(
  kernel void test(int totalN,
      int out_offset,
      int out_dims,
      {% for i=1,numVirtualDims do %}
      int out_dim{{i}},
      int out_stride{{i}},
      {% end %}
      global float*out_data,
      int in1_offset,
      int in1_dims,
      {% for i=1,numVirtualDims do %}
      int in1_dim{{i}},
      int in1_stride{{i}},
      {% end %}
      global float *in1_data,
      int in2_offset,
      int in2_dims,
      {% for i=1,numVirtualDims do %}
      int in2_dim{{i}},
      int in2_stride{{i}},
      {% end %}
      global float *in2_data
      ) {
    int linearId = get_global_id(0);
    if(linearId < totalN) {
      out_data[linearId + out_offset] = in1_data[linearId + in1_offset] * in2_data[linearId + in2_offset]
      {% for i=1,numVirtualDims do %}
      + out_dim{{i}}
      + out_stride{{i}}
      + in1_dim{{i}}
      + in1_stride{{i}}
      + in2_dim{{i}}
      + in2_stride{{i}}
      {% end %}
      ;
    }
  }
)DELIM";

void test(EasyCL *cl, int its, int size, int numVirtualDims) {
  int totalN = size;
  TemplatedKernel kernelBuilder(cl);
  kernelBuilder.set("numVirtualDims", numVirtualDims);
//  cout << kernelBuilder.getRenderedKernel(kernelSource) << endl;
  CLKernel *kernel = kernelBuilder.buildKernel( "apply3flat_" + easycl::toString(numVirtualDims), "apply3flat", kernelSource, "test" );

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
    for(int i = 0; i < numVirtualDims; i++ ) {
      kernel->in(2);
      kernel->in(2);
    }
    kernel->out(outwrap);
    kernel->in(0);
    kernel->in(2);
    for(int i = 0; i < numVirtualDims; i++ ) {
      kernel->in(2);
      kernel->in(2);
    }
    kernel->in(in1wrap);
    kernel->in(0);
    kernel->in(2);
    for(int i = 0; i < numVirtualDims; i++ ) {
      kernel->in(2);
      kernel->in(2);
    }
    kernel->in(in2wrap);

    kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
  }
  cl->finish();
  double end = StatefulTimer::instance()->getSystemMilliseconds();
  cl->dumpProfiling();
  cout << "its=" << its << " size=" << size << " numVirtualDimensions=" << numVirtualDims << " time=" << (end - start) << "ms" << endl;
  outwrap->copyToHost();
  cl->finish();
  if(false){
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
  }

  delete outwrap;
  delete in1wrap;
  delete in2wrap;
  delete[] in1;
  delete[] in2;
  delete[] out;
//  delete kernel;
}

int main(int argc, char *argv[]) {
  int gpu = 0;
  if( argc == 2 ) {
    gpu = atoi(argv[1]);
  }
  cout << "using gpu " << gpu << endl;
  EasyCL *cl = EasyCL::createForIndexedGpu(gpu);
  cl->setProfiling(true);
  for(int it = 0; it < 1; it++ ) {
    int its = it == 1 ? 9000 : 900;
    test(cl, its, 6400, 1);
    test(cl, its, 6400, 2);
    test(cl, its, 6400, 4);
    test(cl, its, 6400, 5);
    test(cl, its, 6400, 10);
    test(cl, its, 6400, 15);
    test(cl, its, 6400, 16);
    test(cl, its, 6400, 20);
    test(cl, its, 6400, 25);
//  test(cl, 9000, 6400, 2);
  }
  cl->dumpProfiling();
  delete cl;
  return 0;
}


