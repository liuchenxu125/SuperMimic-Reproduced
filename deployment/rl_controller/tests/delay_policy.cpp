// Test-only ORT API interposer. Never linked/installed into the controller.
// A test creates SM_TEST_DELAY_POLICY containing "milliseconds calls" AFTER
// startup. A real Run is delayed without blocking ROS sensor/control threads.
#include <onnxruntime_c_api.h>
#include <dlfcn.h>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
static const OrtApi* original_api=nullptr;
static OrtApi wrapped_api;
static OrtApiBase wrapped_base;
static OrtStatus* ORT_API_CALL delayed_run(OrtSession* session, const OrtRunOptions* options,
    const char* const* input_names, const OrtValue* const* inputs, size_t input_count,
    const char* const* output_names, size_t output_count, OrtValue** outputs) noexcept {
  const char* flag=std::getenv("SM_TEST_DELAY_POLICY");
  if(flag) {
    int ms=0,count=0;
    {std::ifstream file(flag); file>>ms>>count;}
    if(ms>0 && ms<=1000 && count>0) {
      if(count==1) std::remove(flag);
      else {std::ofstream file(flag);file<<ms<<' '<<count-1;}
      usleep(ms*1000);
    }
  }
  return original_api->Run(session,options,input_names,inputs,input_count,output_names,output_count,outputs);
}
static const OrtApi* ORT_API_CALL get_wrapped_api(uint32_t version) noexcept {
  using Getter=const OrtApiBase* (*)();
  const auto real=reinterpret_cast<Getter>(dlsym(RTLD_NEXT,"OrtGetApiBase"));
  original_api=real()->GetApi(version);
  if(!original_api) return nullptr;
  wrapped_api=*original_api;wrapped_api.Run=delayed_run;return &wrapped_api;
}
extern "C" const OrtApiBase* ORT_API_CALL OrtGetApiBase(void) noexcept {
  using Getter=const OrtApiBase* (*)();
  const auto real=reinterpret_cast<Getter>(dlsym(RTLD_NEXT,"OrtGetApiBase"));
  wrapped_base=*real();wrapped_base.GetApi=get_wrapped_api;return &wrapped_base;
}
