// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License

// Manual integration test: verifies whether issuing a second STS token
// invalidates the first one on HuaweiCloud IAM.
//
// Requires env vars:
//   HUAWEICLOUD_SDK_REGION, HUAWEICLOUD_SDK_PROJECT_ID,
//   HUAWEICLOUD_SDK_IDP_ID, HUAWEICLOUD_SDK_ID_TOKEN_FILE,
//   MILVUS_OBS_BUCKET
//
// Build and run:
//   cd cmake_build && make HuaweiCloudSTSConflictTest -j4
//   ./output/unittest/HuaweiCloudSTSConflictTest

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentials.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/platform/Environment.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/logging/LogMacros.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/ListObjectsV2Request.h>
#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "HuaweiCloudCredentialsProvider.h"

static const char* LOG_TAG = "STSConflictTest";

class HuaweiCloudSTSConflictTest : public ::testing::Test {
 protected:
    static void
    SetUpTestSuite() {
        Aws::SDKOptions options;
        options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Info;
        Aws::InitAPI(options);
    }

    static void
    TearDownTestSuite() {
        Aws::SDKOptions options;
        Aws::ShutdownAPI(options);
    }

    void
    SetUp() override {
        region_ = Aws::Environment::GetEnv("HUAWEICLOUD_SDK_REGION");
        bucket_ = Aws::Environment::GetEnv("MILVUS_OBS_BUCKET");

        if (region_.empty() || bucket_.empty()) {
            GTEST_SKIP() << "Skipping: HUAWEICLOUD_SDK_REGION or "
                            "MILVUS_OBS_BUCKET not set";
        }

        auto tokenFile =
            Aws::Environment::GetEnv("HUAWEICLOUD_SDK_ID_TOKEN_FILE");
        if (tokenFile.empty()) {
            GTEST_SKIP()
                << "Skipping: HUAWEICLOUD_SDK_ID_TOKEN_FILE not set";
        }
    }

    // Try listing 1 object in OBS using the given credentials
    bool
    tryListObjects(const Aws::Auth::AWSCredentials& creds,
                   std::string& errorMsg) {
        Aws::Client::ClientConfiguration config;
        config.region = region_;
        config.scheme = Aws::Http::Scheme::HTTPS;
        config.endpointOverride =
            "obs." + region_ + ".myhuaweicloud.com";

        auto s3Client = Aws::MakeShared<Aws::S3::S3Client>(
            LOG_TAG, creds, config,
            Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
            /*useVirtualAddressing=*/true);

        Aws::S3::Model::ListObjectsV2Request request;
        request.SetBucket(bucket_);
        request.SetMaxKeys(1);

        auto outcome = s3Client->ListObjectsV2(request);
        if (!outcome.IsSuccess()) {
            errorMsg = outcome.GetError().GetMessage();
            return false;
        }
        return true;
    }

    Aws::String region_;
    Aws::String bucket_;
};

// Access the private credential provider internals for testing
class STSConflictTestHelper {
 public:
    static Aws::Auth::AWSCredentials
    getCredentials(
        Aws::Auth::
            HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider& p) {
        // Force a fresh load by calling GetAWSCredentials (which calls
        // RefreshIfExpired → Reload on first call)
        return p.GetAWSCredentials();
    }

    static void
    forceReload(
        Aws::Auth::
            HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider& p) {
        p.Reload();
    }
};

TEST_F(HuaweiCloudSTSConflictTest, GoStyleSTS_TwoTokens) {
    // This test simulates two Go-style STS calls (CreateTemporaryAccessKeyByToken)
    // by creating two independent credential providers and forcing reload on each.

    Aws::Auth::HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider
        provider1;
    auto creds1 = STSConflictTestHelper::getCredentials(provider1);

    if (creds1.IsEmpty()) {
        GTEST_SKIP() << "Could not get token1 (IAM issue?), skipping";
    }

    std::cout << "Token1 AK: "
              << creds1.GetAWSAccessKeyId().substr(0, 8) << "..."
              << std::endl;

    // Verify token1 works
    std::string err;
    ASSERT_TRUE(tryListObjects(creds1, err))
        << "Token1 failed immediately: " << err;
    std::cout << "Token1 works immediately after issuance ✓" << std::endl;

    // Small gap
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Get token2 from a separate provider instance (simulating independent STS call)
    Aws::Auth::HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider
        provider2;
    auto creds2 = STSConflictTestHelper::getCredentials(provider2);

    if (creds2.IsEmpty()) {
        GTEST_SKIP() << "Could not get token2 (IAM issue?), skipping";
    }

    std::cout << "Token2 AK: "
              << creds2.GetAWSAccessKeyId().substr(0, 8) << "..."
              << std::endl;

    if (creds1.GetAWSAccessKeyId() == creds2.GetAWSAccessKeyId()) {
        std::cout << "WARNING: token1 and token2 have the same AK"
                  << std::endl;
    } else {
        std::cout << "Token1 and Token2 have different AKs ✓" << std::endl;
    }

    // Verify token2 works
    ASSERT_TRUE(tryListObjects(creds2, err))
        << "Token2 failed: " << err;
    std::cout << "Token2 works ✓" << std::endl;

    // --- THE KEY TEST ---
    bool token1StillWorks = tryListObjects(creds1, err);
    if (!token1StillWorks) {
        std::cout << "TOKEN CONFLICT DETECTED: token1 is INVALID after "
                     "token2 was issued: "
                  << err << std::endl;
        std::cout << "CONCLUSION: HuaweiCloud IAM invalidates previous "
                     "STS tokens when new ones are issued"
                  << std::endl;
        FAIL() << "Token1 invalidated by Token2: " << err;
    } else {
        std::cout << "Token1 still works after Token2 was issued ✓"
                  << std::endl;
        std::cout << "CONCLUSION: STS tokens coexist — no mutual "
                     "invalidation"
                  << std::endl;
    }
}

TEST_F(HuaweiCloudSTSConflictTest, CrossLayerConflict_CppThenCpp) {
    // Simulates Go+C++ cross-layer scenario:
    // both layers call securitytokens endpoint independently.
    // Here we just do two sequential calls from C++ to check for conflict.

    Aws::Auth::HuaweiCloudSTSAssumeRoleWebIdentityCredentialsProvider
        provider;

    // First call
    auto creds1 = STSConflictTestHelper::getCredentials(provider);
    if (creds1.IsEmpty()) {
        GTEST_SKIP() << "Could not get token1, skipping";
    }
    std::cout << "Token1 AK: "
              << creds1.GetAWSAccessKeyId().substr(0, 8) << "..."
              << std::endl;

    std::string err;
    ASSERT_TRUE(tryListObjects(creds1, err))
        << "Token1 failed: " << err;
    std::cout << "Token1 works ✓" << std::endl;

    // Force second STS call on SAME provider
    std::this_thread::sleep_for(std::chrono::seconds(2));
    STSConflictTestHelper::forceReload(provider);
    auto creds2 = STSConflictTestHelper::getCredentials(provider);

    if (creds2.IsEmpty()) {
        GTEST_SKIP() << "Could not get token2, skipping";
    }
    std::cout << "Token2 AK: "
              << creds2.GetAWSAccessKeyId().substr(0, 8) << "..."
              << std::endl;

    ASSERT_TRUE(tryListObjects(creds2, err))
        << "Token2 failed: " << err;
    std::cout << "Token2 works ✓" << std::endl;

    // --- THE KEY TEST: does token1 still work? ---
    bool token1StillWorks = tryListObjects(creds1, err);
    if (!token1StillWorks) {
        std::cout << "TOKEN CONFLICT DETECTED on same provider: " << err
                  << std::endl;
        FAIL() << "Token1 invalidated by Token2 on same provider: " << err;
    } else {
        std::cout
            << "Token1 still works after forced reload on same provider ✓"
            << std::endl;
    }
}
