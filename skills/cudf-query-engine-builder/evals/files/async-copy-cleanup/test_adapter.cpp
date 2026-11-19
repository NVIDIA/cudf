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
void expect_answer(Answer const& result, std::int64_t sum, std::int64_t count) {
  require(result.sum==sum && result.count==count, "sum/count differ from independent reference");
  require(result.origin==Origin::backend, "answer was not produced by requested backend");
}
void check_failure(Fault fault) {
  Backend backend; backend.fault=fault;
  bool caught=false;
  try { (void)execute(backend, backend.input({1,std::nullopt,4,8,-2}),3); }
  catch (std::bad_alloc const&) { caught=true; }
  require(caught,"original allocation error was not propagated");
  require(clean(*backend.audit),audit_state(*backend.audit));
  backend.fault={}; backend.allocations=0;
  auto before=backend.audit->backend_operations;
  expect_answer(execute(backend,backend.input({7,11,std::nullopt,-1}),5),18,2);
  require(backend.audit->backend_operations-before==2,"recovery call did not execute upload and backend operator");
  require(backend.audit->cpu_operations==0,"CPU fallback was used without permission");
  require(clean(*backend.audit),audit_state(*backend.audit));
}
int main() {
  check("nullable_normal_result",[] {
    Backend b; expect_answer(execute(b,b.input({1,std::nullopt,4,8,-2}),3),12,2);
    require(b.audit->backend_operations==2,"upload and backend operator required");
    require(b.audit->cpu_operations==0,"unexpected CPU fallback");
    require(clean(*b.audit),audit_state(*b.audit));
  });
  check("empty_result",[] { Backend b; expect_answer(execute(b,b.input({}),0),0,0); require(clean(*b.audit),audit_state(*b.audit)); });
  check("all_null_result",[] { Backend b; expect_answer(execute(b,b.input({std::nullopt,std::nullopt}),-9),0,0); require(clean(*b.audit),audit_state(*b.audit)); });
  check("allocation_before_submission_and_recovery",[] { check_failure({1,false}); });
  check("output_allocation_failure_and_recovery",[] { check_failure({2,false}); });
  check("result_wrapper_failure_and_recovery",[] { check_failure({0,true}); });
  return report();
}
