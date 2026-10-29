/* Test-only LD_PRELOAD shim. Never built/installed into the deployment package.
 * Interpose the C rcl_publish ABI without linking a second ROS runtime.
 * Only isolated test topics are eligible; an atomic flag removal arms one stall.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
struct rcl_publisher_s;
struct rmw_publisher_allocation_s;
int rcl_publish(const struct rcl_publisher_s *publisher, const void *message,
                struct rmw_publisher_allocation_s *allocation) {
  typedef int (*Publish)(const struct rcl_publisher_s *, const void *, struct rmw_publisher_allocation_s *);
  typedef const char *(*Topic)(const struct rcl_publisher_s *);
  Publish real_publish = (Publish)dlsym(RTLD_NEXT, "rcl_publish");
  Topic topic_name = (Topic)dlsym(RTLD_NEXT, "rcl_publisher_get_topic_name");
  const char *topic = topic_name(publisher);
  const char *flag = NULL;
  if (strcmp(topic, "/supermimic_offline_test/command") == 0 ||
      strcmp(topic, "/supermimic_controller/dry_run_command") == 0)
    flag = getenv("SM_TEST_BLOCK_COMMAND");
  if (strcmp(topic, "/supermimic_controller/status") == 0)
    flag = getenv("SM_TEST_BLOCK_STATUS");
  if (flag && unlink(flag) == 0) usleep(350000);
  return real_publish(publisher, message, allocation);
}
