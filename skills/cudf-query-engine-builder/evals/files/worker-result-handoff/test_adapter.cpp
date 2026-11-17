// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "adapter.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace fixture;
struct Check { std::string name; bool passed; std::string detail; };
std::vector<Check> checks;
void require(bool condition, std::string const& message) { if (!condition) throw std::runtime_error(message); }
std::string audit_state(Audit const& a) {
  std::ostringstream s;
  s << "early_release=" << a.early_release << ", allocator_lifetime_error=" << a.allocator_lifetime_error
    << ", stream_abandoned=" << a.stream_abandoned << ", live_buffers=" << a.live_buffers
    << ", pending_uses=" << a.pending_uses << ", asynchronous_errors=" << a.asynchronous_errors;
  return s.str();
}
template<class F> void check(std::string name, F fn) {
  try { fn(); checks.push_back({name,true,"checks satisfied"}); }
  catch (std::exception const& e) { checks.push_back({name,false,e.what()}); }
  catch (...) { checks.push_back({name,false,"unexpected non-standard exception"}); }
}
int report() {
  bool passed = true;
  for (auto const& c : checks) passed = passed && c.passed;
  std::cout << "{\"backend\":\"deterministic-cpu-simulator\",\"native_execution\":false,\"passed\":"
            << (passed ? "true" : "false") << ",\"checks\":[";
  for (std::size_t i=0; i<checks.size(); ++i) {
    if (i) std::cout << ',';
    auto const& c=checks[i];
    std::cout << "{\"name\":" << std::quoted(c.name) << ",\"passed\":" << (c.passed ? "true" : "false")
              << ",\"detail\":" << std::quoted(c.detail) << '}';
  }
  std::cout << "]}\n";
  return passed ? 0 : 1;
}
void expect_answer(Answer const& value,std::int64_t sum,std::int64_t count,Origin origin) {
  require(value.sum==sum && value.count==count,"sum/count differ from independent reference");
  require(value.origin==origin,"reported backend origin does not match requested path");
}
void normal(Rows rows,std::int64_t cutoff,std::int64_t sum,std::int64_t count) {
  Backend b;
  auto result=submit(b,b.input(std::move(rows)),b.stream(),cutoff,true);
  require(b.audit->backend_operations==0,"submit must return before running queued backend work");
  require(b.audit->pending_uses>0,"submit did not queue the requested backend work");
  auto value=result->get();
  expect_answer(value,sum,count,Origin::backend);
  expect_answer(result->get(),sum,count,Origin::backend);
  require(b.audit->backend_operations==2,"upload and backend operator must each execute exactly once");
  require(b.audit->cpu_operations==0,"backend request silently used CPU fallback");
  result.reset();
  require(clean(*b.audit),audit_state(*b.audit));
}
int main() {
  check("worker_handoff_after_caller_owners_expire",[] { normal({1,std::nullopt,4,8,-2},3,12,2); });
  check("empty_worker_result",[] { normal({},0,0,0); });
  check("all_null_worker_result",[] { normal({std::nullopt,std::nullopt},-10,0,0); });
  check("discard_before_get",[] {
    Backend b;
    auto result=submit(b,b.input({2,5,9}),b.stream(),3,true);
    require(b.audit->backend_operations==0,"submit synchronously executed backend work");
    result.reset();
    require(clean(*b.audit),audit_state(*b.audit));
    require(b.audit->backend_operations==2,"discard abandoned work instead of completing owned uses");
  });
  check("two_outstanding_results",[] {
    Backend b;
    auto first=submit(b,b.input({3,9}),b.stream(),4,true);
    auto second=submit(b,b.input({4,6}),b.stream(),2,true);
    expect_answer(second->get(),10,2,Origin::backend);
    expect_answer(first->get(),9,1,Origin::backend);
    first.reset(); second.reset();
    require(b.audit->backend_operations==4,"outstanding requests must execute independent backend work");
    require(clean(*b.audit),audit_state(*b.audit));
  });
  check("explicit_cpu_fallback_is_labeled",[] {
    Backend b;
    auto result=submit(b,b.input({2,5,9}),b.stream(),3,false);
    expect_answer(result->get(),14,2,Origin::cpu);
    require(b.audit->backend_operations==0 && b.audit->cpu_operations==1,"fallback counters do not match actual work");
    result.reset(); require(clean(*b.audit),audit_state(*b.audit));
  });
  return report();
}
