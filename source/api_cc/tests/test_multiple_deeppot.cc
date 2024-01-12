// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "DeepPot.h"


TEST(TestMultipleDeepPot, multiple_deep_pot) {
  std::string file_name = "../../tests/infer/deeppot.pbtxt";
  deepmd::convert_pbtxt_to_pb("../../tests/infer/deeppot.pbtxt",
                                "deeppot.pb");

  deepmd::DeepPot  dp1("deeppot.pb");
  deepmd::DeepPot  dp2("deeppot.pb");

  remove("deeppot.pb");
};