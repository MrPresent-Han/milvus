#pragma once

#include <aws/core/internal/AWSHttpResourceClient.h>

namespace Aws {
namespace Http {
class HttpClient;
class HttpRequest;
enum class HttpResponseCode;
}  // namespace Http

namespace Internal {
class AWS_CORE_API HuaweiCloudSTSCredentialsClient : public AWSHttpResourceClient {
 public:
    explicit HuaweiCloudSTSCredentialsClient(const Aws::Client::ClientConfiguration& clientConfiguration);

    HuaweiCloudSTSCredentialsClient&
    operator=(HuaweiCloudSTSCredentialsClient& rhs) = delete;
    HuaweiCloudSTSCredentialsClient(
        const HuaweiCloudSTSCredentialsClient& rhs) = delete;
    HuaweiCloudSTSCredentialsClient&
    operator=(HuaweiCloudSTSCredentialsClient&& rhs) = delete;
    HuaweiCloudSTSCredentialsClient(
        const HuaweiCloudSTSCredentialsClient&& rhs) = delete;

    struct STSAssumeRoleWithWebIdentityRequest {
        Aws::String region;
        Aws::String providerId;
        Aws::String webIdentityToken;
        Aws::String roleArn;
        Aws::String roleSessionName;
    };
    
    struct STSAssumeRoleWithWebIdentityResult {
        Aws::Auth::AWSCredentials creds;
    };

    STSAssumeRoleWithWebIdentityResult
    GetAssumeRoleWithWebIdentityCredentials(
        const STSAssumeRoleWithWebIdentityRequest& request);

 private:
    Aws::String m_endpoint;
    
    // 内部结构体用于STS调用结果
    struct STSCallResult {
        bool success;
        Aws::Auth::AWSCredentials credentials;
        Aws::String errorMessage;
    };
    
    // 第二步：使用用户token调用华为云STS API获取临时凭证
    STSCallResult callHuaweiCloudSTS(const Aws::String& userToken, const STSAssumeRoleWithWebIdentityRequest& request);

};
}  // namespace Internal
}  // namespace Aws