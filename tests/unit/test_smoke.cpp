#include "tinycu/version.h"

#include <gtest/gtest.h>

#include <string>

TEST(Smoke, Arithmetic) { EXPECT_EQ(1 + 1, 2); }

TEST(Smoke, Version) { EXPECT_EQ(std::string(tinycu::version()), "0.0.1"); }
