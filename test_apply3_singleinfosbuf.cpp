#include <iostream>
using namespace std;
#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/StatefulTimer.h"
#include "util/easycl_stringhelper.h"

typedef struct Info {
  int dims;
  int offset;
  int sizes[5];
  int strides[5];
} Info;

static const char *kernelSource = R"DELIM(
  typedef struct Info {
    int dims;
    int offset;
    int sizes[5];
    int strides[5];
  } Info;

  kernel void test(int totalN,
      global struct Info *infos,
      int out_idx,
      int in1_idx,
      int in2_idx,
      global float*out_data,
      global float *in1_data,
      global float *in2_data
      ) {
    global struct Info *out_info = &infos[out_idx];
    global struct Info *in1_info = &infos[in1_idx];
    global struct Info *in2_info =  &infos[in2_idx];
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

  #define MAX_INFOS 128
  Info infos[MAX_INFOS];
  Info *outInfo = &infos[0];
  Info *in1Info = &infos[1];
  Info *in2Info = &infos[2];
  outInfo->offset = in1Info->offset = in2Info->offset = 0;
  outInfo->dims = in1Info->dims = in2Info->dims = 1;
  outInfo->sizes[0] = in1Info->sizes[0] = in2Info->sizes[0] = 6400;
  outInfo->strides[0] = in1Info->strides[0] = in2Info->strides[0] = 1;

  float *infosFloat = reinterpret_cast<float *>(&infos[0]);
  CLWrapper *infosWrap = cl->wrap((sizeof(Info)*3 + sizeof(float)-1)/sizeof(float), infosFloat);
  infosWrap->copyToDevice();

  Info infosb[3];
  Info *outInfob = &infosb[0];
  Info *in1Infob = &infosb[1];
  Info *in2Infob = &infosb[2];
  outInfob->offset = in1Infob->offset = in2Infob->offset = 0;
  outInfob->dims = in1Infob->dims = in2Infob->dims = 1;
  outInfob->sizes[0] = in1Infob->sizes[0] = in2Infob->sizes[0] = 6400;
  outInfob->strides[0] = in1Infob->strides[0] = in2Infob->strides[0] = 1;

  float *infosFloatb = reinterpret_cast<float *>(&infosb[0]);
  CLWrapper *infosWrapb = cl->wrap((sizeof(Info)*3 + sizeof(float)-1)/sizeof(float), infosFloatb);
  infosWrapb->copyToDevice();

  typedef struct InfosStruct {
      Info infos[3];
      CLWrapper *wrapper;
  } InfosStruct;

  vector< InfosStruct* > infosStore;
  InfosStruct *is = 0;
  for( int i = 0; i < 60; i++ ) { // add some dummy ones, to pretend we are running char-rnn
    InfosStruct *infosStruct = new InfosStruct();
    is = infosStruct;
    for( int i = 0; i < 3; i++ ) {
      is->infos[i].offset = 0;
      is->infos[i].dims = 1;
      is->infos[i].sizes[0] = 6400;
      is->infos[i].strides[0] = 2; // just this last value will be different :-)
    }
    infosStruct->wrapper = cl->wrap((sizeof(Info)*3 + sizeof(float)-1)/sizeof(float), reinterpret_cast< float *>(&infosStruct->infos[0]));
    infosStruct->wrapper->copyToDevice();
    infosStore.push_back(infosStruct);
  }
  is = infosStore[59];
  for( int i = 0; i < 3; i++ ) {
    is->infos[i].offset = 0;
    is->infos[i].dims = 1;
    is->infos[i].sizes[0] = 6400;
    is->infos[i].strides[0] = 1;
  }

  cl->finish();
  cl->dumpProfiling();

  double start = StatefulTimer::instance()->getSystemMilliseconds();
  for(int it = 0; it < its; it++) {
    kernel->in(totalN);

    is = 0;
    for( int i = 0; i < (int)infosStore.size(); i++ ) {
      InfosStruct *tis = infosStore[i];
      bool possiblematch = true;
      for( int i = 0; i < 3; i++ ) {
        if(tis->infos[i].offset != 0) {
          possiblematch = false;
          break;
        }
        if(tis->infos[i].dims != 1) {
          possiblematch = false;
          break;
        }
        if(tis->infos[i].sizes[0] != 6400) {
          possiblematch = false;
          break;
        }
        if(tis->infos[i].strides[0] != 1) {
          possiblematch = false;
          break;
        }
      }
      if(possiblematch) {
        is = tis;
        break;
      }
    }
    if(is == 0) {
      cout << "no mathcin is found" << endl;
    }

    kernel->in(is->wrapper);

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
  test(cl, 900, 6400, true);
  test(cl, 9000, 6400, true);
  cl->dumpProfiling();
  delete cl;
  return 0;
}


