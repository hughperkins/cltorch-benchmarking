#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

typedef struct Info {
  int dims;
  int offset;
  int sizes[25];
  int strides[25];
} Info;

static const char *kernelSource = R"DELIM(
  typedef struct Info {
    int dims;
    int offset;
    int sizes[25];
    int strides[25];
  } Info;

  kernel void test(int totalN,
      global struct Info *out_info,
      global struct Info *in1_info,
      global struct Info *in2_info,
      global float*out_data,
      global float *in1_data,
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

  Info outInfo;
  Info in1Info;
  Info in2Info;
  outInfo.offset = in1Info.offset = in2Info.offset = 0;
  outInfo.dims = in1Info.dims = in2Info.dims = 1;
  outInfo.sizes[0] = in1Info.sizes[0] = in2Info.sizes[0] = 6400;
  outInfo.strides[0] = in1Info.strides[0] = in2Info.strides[0] = 1;

  float *outInfoFloat = reinterpret_cast<float *>(&outInfo);
  float *in1InfoFloat = reinterpret_cast<float *>(&in1Info);
  float *in2InfoFloat = reinterpret_cast<float *>(&in2Info);
  CLWrapper *outInfoWrap = cl->wrap(64, outInfoFloat);
  CLWrapper *in1InfoWrap = cl->wrap(64, in1InfoFloat);
  CLWrapper *in2InfoWrap = cl->wrap(64, in2InfoFloat);
  outInfoWrap->copyToDevice();
  in1InfoWrap->copyToDevice();
  in2InfoWrap->copyToDevice();

  cl->finish();
  cl->dumpProfiling();

  double start = StatefulTimer::instance()->getSystemMilliseconds();
  for(int it = 0; it < its; it++) {
    kernel->in(totalN);

    if(reuseStructBuffers) {
      kernel->in(outInfoWrap);
      kernel->in(in1InfoWrap);
      kernel->in(in2InfoWrap);
    } else {
      kernel->in(1, &outInfo);
      kernel->in(1, &in1Info);
      kernel->in(1, &in2Info);
    }

    kernel->out(outwrap);
    kernel->in(in1wrap);
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
  test(cl, 900, 6400, false);
  test(cl, 900, 6400, true);
  test(cl, 9000, 6400, false);
  test(cl, 9000, 6400, true);
  cl->dumpProfiling();
  delete cl;
  return 0;
}


