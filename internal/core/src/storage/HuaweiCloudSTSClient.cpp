#include "HuaweiCloudSTSClient.h"
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpResponse.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/utils/DateTime.h>


namespace Aws {
namespace Http {
class HttpClient;
class HttpRequest;
enum class HttpResponseCode;
}  // namespace Http

namespace Internal {

static const char STS_RESOURCE_CLIENT_LOG_TAG[] =
    "HuaweiCloudSTSResourceClient";

HuaweiCloudSTSCredentialsClient::HuaweiCloudSTSCredentialsClient(
    const Aws::Client::ClientConfiguration& clientConfiguration)
    : AWSHttpResourceClient(clientConfiguration, STS_RESOURCE_CLIENT_LOG_TAG) {
    SetErrorMarshaller(Aws::MakeUnique<Aws::Client::XmlErrorMarshaller>(
        STS_RESOURCE_CLIENT_LOG_TAG));

    m_endpoint = "https://iam.{region}.myhuaweicloud.com/v3.0/OS-AUTH/id-token/tokens";
    AWS_LOGSTREAM_INFO(
        STS_RESOURCE_CLIENT_LOG_TAG,
        "Creating STS ResourceClient with endpoint: " << m_endpoint);
}

HuaweiCloudSTSCredentialsClient::STSAssumeRoleWithWebIdentityResult
HuaweiCloudSTSCredentialsClient::GetAssumeRoleWithWebIdentityCredentials(
    const STSAssumeRoleWithWebIdentityRequest& request) {
    // Calculate query string
    Aws::StringStream ss;
    
    // 构建请求JSON，使用华为云IAM格式
    ss << R"({ 
   "auth" : { 
     "id_token" : { 
       "id" : ")" << request.webIdentityToken << R"("
     }, 
     "scope": { 
       "project" : { 
         "id" : ")" << request.roleArn << R"(",
         "name" : ")" << request.region << R"("
       } 
     } 
   } 
 })";

    // 构建完整的端点URL，替换region占位符
    Aws::String endpoint = m_endpoint;
    size_t pos = endpoint.find("{region}");
    if (pos != Aws::String::npos) {
        endpoint.replace(pos, 8, request.region);  // 8 是 "{region}" 的长度
    }

    std::shared_ptr<Aws::Http::HttpRequest> httpRequest(
        Aws::Http::CreateHttpRequest(
            endpoint,
            Aws::Http::HttpMethod::HTTP_POST,
            Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));

    httpRequest->SetUserAgent(Aws::Client::ComputeUserAgentString());
    httpRequest->SetHeaderValue("X-Idp-Id", request.providerId);
    httpRequest->SetHeaderValue("Content-Type", "application/json");
    
    // 构建请求体
    std::shared_ptr<Aws::IOStream> body =
        Aws::MakeShared<Aws::StringStream>("STS_RESOURCE_CLIENT_LOG_TAG");
    
    std::cout << "hc===body: " << ss.str() << std::endl;
    *body << ss.str();
    httpRequest->AddContentBody(body);
    body->seekg(0, body->end);
    auto streamSize = body->tellg();
    body->seekg(0, body->beg);
    Aws::StringStream contentLength;
    contentLength << streamSize;
    httpRequest->SetContentLength(contentLength.str());
    httpRequest->SetContentType("application/json");

    auto headers = httpRequest->GetHeaders();
    
    // 调试信息：打印请求详情
    std::cout << "=== HTTP Request Debug Info ===" << std::endl;
    std::cout << "Endpoint: " << endpoint << std::endl;
    std::cout << "Method: POST" << std::endl;
    std::cout << "Request Body: " << ss.str() << std::endl;
    std::cout << "Headers:" << std::endl;
    for (const auto& header : headers) {
        std::cout << "  " << header.first << ": " << header.second << std::endl;
    }
    std::cout << "===============================" << std::endl;
    
    // 发送请求并获取响应
    std::cout << "Sending HTTP request..." << std::endl;
    auto awsResult = GetResourceWithAWSWebServiceResult(httpRequest);
    
    // 调试信息：打印响应详情
    std::cout << "=== HTTP Response Debug Info ===" << std::endl;
    
    // 检查响应是否成功
    if (awsResult.GetResponseCode() != Aws::Http::HttpResponseCode::NO_RESPONSE) {
        std::cout << "Response Code: " << static_cast<int>(awsResult.GetResponseCode()) << std::endl;
    } else {
        std::cout << "Response Code: NO_RESPONSE (connection failed?)" << std::endl;
    }
    
    // 获取响应体
    Aws::String credentialsStr = awsResult.GetPayload();
    std::cout << "Response Body Length: " << credentialsStr.length() << std::endl;
    std::cout << "Response Body: " << credentialsStr << std::endl;
    
    // 获取响应头
    auto responseHeaders = awsResult.GetHeaderValueCollection();
    std::cout << "Response Headers:" << std::endl;
    for (const auto& header : responseHeaders) {
        std::cout << "  " << header.first << ": " << header.second << std::endl;
    }
    
    // 华为云IAM特殊处理：token信息在x-subject-token响应头中
    auto subjectTokenIter = responseHeaders.find("x-subject-token");
    if (subjectTokenIter != responseHeaders.end()) {
        std::cout << "Found x-subject-token in response headers!" << std::endl;
        std::cout << "x-subject-token length: " << subjectTokenIter->second.length() << std::endl;
        
        // 华为云的token实际上是JWT格式，包含在x-subject-token头中
        // 我们需要解析这个JWT来获取凭证信息
        credentialsStr = subjectTokenIter->second;
        std::cout << "Using x-subject-token as credentials: " << credentialsStr.substr(0, 100) << "..." << std::endl;
    }
    
    std::cout << "================================" << std::endl;

    STSAssumeRoleWithWebIdentityResult result;
    if (credentialsStr.empty()) {
        AWS_LOGSTREAM_WARN(STS_RESOURCE_CLIENT_LOG_TAG,
                           "Get an empty credential from Huawei Cloud IAM");
        return result;
    }

    // 华为云IAM返回的是JWT token在x-subject-token头中
    // 对于华为云，我们需要解析JWT token来获取凭证信息
    // 但是对于临时凭证，华为云通常会在响应体中返回JSON格式的凭证
    
    // 首先尝试解析响应体中的JSON（如果有的话）
    if (!credentialsStr.empty() && credentialsStr[0] == '{') {
        std::cout << "Parsing JSON response body..." << std::endl;
        auto json = Utils::Json::JsonView(credentialsStr);
        auto tokenNode = json.GetObject("token");
        if (!tokenNode.IsNull()) {
            auto credentialsNode = tokenNode.GetObject("credential");
            if (!credentialsNode.IsNull()) {
                result.creds.SetAWSAccessKeyId(credentialsNode.GetString("access"));
                result.creds.SetAWSSecretKey(credentialsNode.GetString("secret"));
                result.creds.SetSessionToken(credentialsNode.GetString("securitytoken"));
                
                auto expiresAt = tokenNode.GetString("expires_at");
                if (!expiresAt.empty()) {
                    result.creds.SetExpiration(Aws::Utils::DateTime(
                        Aws::Utils::StringUtils::Trim(expiresAt.c_str()).c_str(),
                        Aws::Utils::DateFormat::ISO_8601));
                }
                std::cout << "Successfully parsed credentials from JSON response" << std::endl;
            }
        }
    } else {
        // 如果响应体为空或不是JSON，说明华为云IAM使用了不同的认证流程
        // x-subject-token包含的是用户认证token，需要进行第二步：调用STS API获取临时凭证
        std::cout << "Response body is empty or not JSON format" << std::endl;
        std::cout << "Starting Step 2: Call STS API to get temporary credentials" << std::endl;
        
        if (!credentialsStr.empty()) {
            // 第二步：使用x-subject-token调用华为云STS API获取临时凭证
            auto stsResult = callHuaweiCloudSTS(credentialsStr, request);
            if (stsResult.success) {
                result.creds = stsResult.credentials;
                std::cout << "Successfully obtained temporary credentials from STS" << std::endl;
            } else {
                std::cout << "Failed to obtain temporary credentials from STS: " << stsResult.errorMessage << std::endl;
            }
        }
    }

    return result;
}

HuaweiCloudSTSCredentialsClient::STSCallResult
HuaweiCloudSTSCredentialsClient::callHuaweiCloudSTS(
    const Aws::String& userToken, 
    const STSAssumeRoleWithWebIdentityRequest& request) {
    
    STSCallResult result;
    result.success = false;
    
    std::cout << "=== Step 2: Calling Huawei Cloud STS API ===" << std::endl;
    
    // 华为云STS API端点
    Aws::String stsEndpoint = "https://sts." + request.region + ".myhuaweicloud.com";
    std::cout << "STS Endpoint: " << stsEndpoint << std::endl;
    
    // 构建STS请求体 - AssumeRoleWithWebIdentity
    Aws::StringStream stsRequestBody;
    stsRequestBody << R"({
    "auth": {
        "identity": {
            "methods": ["token"],
            "token": {
                "id": ")" << userToken << R"("
            }
        },
        "scope": {
            "project": {
                "id": ")" << request.roleArn << R"("
            }
        }
    }
})";

    std::cout << "STS Request Body: " << stsRequestBody.str() << std::endl;
    
    // 创建HTTP请求
    std::shared_ptr<Aws::Http::HttpRequest> stsHttpRequest(
        Aws::Http::CreateHttpRequest(
            stsEndpoint + "/v3/auth/tokens",
            Aws::Http::HttpMethod::HTTP_POST,
            Aws::Utils::Stream::DefaultResponseStreamFactoryMethod));
    
    stsHttpRequest->SetUserAgent(Aws::Client::ComputeUserAgentString());
    stsHttpRequest->SetHeaderValue("Content-Type", "application/json");
    stsHttpRequest->SetHeaderValue("X-Auth-Token", userToken);
    
    // 构建请求体
    std::shared_ptr<Aws::IOStream> stsBody =
        Aws::MakeShared<Aws::StringStream>("STS_RESOURCE_CLIENT_LOG_TAG");
    *stsBody << stsRequestBody.str();
    
    stsHttpRequest->AddContentBody(stsBody);
    stsBody->seekg(0, stsBody->end);
    auto stsStreamSize = stsBody->tellg();
    stsBody->seekg(0, stsBody->beg);
    Aws::StringStream stsContentLength;
    stsContentLength << stsStreamSize;
    stsHttpRequest->SetContentLength(stsContentLength.str());
    
    std::cout << "Sending STS request..." << std::endl;
    
    try {
        // 发送STS请求
        auto stsAwsResult = GetResourceWithAWSWebServiceResult(stsHttpRequest);
        
        std::cout << "STS Response Code: " << static_cast<int>(stsAwsResult.GetResponseCode()) << std::endl;
        
        // 获取STS响应
        Aws::String stsResponseBody = stsAwsResult.GetPayload();
        std::cout << "STS Response Body Length: " << stsResponseBody.length() << std::endl;
        std::cout << "STS Response Body: " << stsResponseBody << std::endl;
        
        // 获取STS响应头
        auto stsResponseHeaders = stsAwsResult.GetHeaderValueCollection();
        std::cout << "STS Response Headers:" << std::endl;
        for (const auto& header : stsResponseHeaders) {
            std::cout << "  " << header.first << ": " << header.second << std::endl;
        }
        
        // 解析STS响应获取临时凭证
        if (stsAwsResult.GetResponseCode() == Aws::Http::HttpResponseCode::CREATED || 
            stsAwsResult.GetResponseCode() == Aws::Http::HttpResponseCode::OK) {
            
            // 检查是否有临时凭证在响应体中
            if (!stsResponseBody.empty()) {
                auto stsJson = Utils::Json::JsonView(stsResponseBody);
                auto tokenNode = stsJson.GetObject("token");
                if (!tokenNode.IsNull()) {
                    // 查找临时凭证
                    auto credentialsNode = tokenNode.GetObject("credential");
                    if (!credentialsNode.IsNull()) {
                        result.credentials.SetAWSAccessKeyId(credentialsNode.GetString("access"));
                        result.credentials.SetAWSSecretKey(credentialsNode.GetString("secret"));
                        result.credentials.SetSessionToken(credentialsNode.GetString("securitytoken"));
                        
                        auto expiresAt = tokenNode.GetString("expires_at");
                        if (!expiresAt.empty()) {
                            result.credentials.SetExpiration(Aws::Utils::DateTime(
                                Aws::Utils::StringUtils::Trim(expiresAt.c_str()).c_str(),
                                Aws::Utils::DateFormat::ISO_8601));
                        }
                        
                        result.success = true;
                        std::cout << "Successfully parsed temporary credentials from STS response" << std::endl;
                    } else {
                        result.errorMessage = "No credential object found in STS response";
                    }
                } else {
                    result.errorMessage = "No token object found in STS response";
                }
            } else {
                // 如果响应体为空，检查响应头中是否有临时凭证信息
                auto stsSubjectTokenIter = stsResponseHeaders.find("x-subject-token");
                if (stsSubjectTokenIter != stsResponseHeaders.end()) {
                    // 使用新的token作为session token
                    result.credentials.SetSessionToken(stsSubjectTokenIter->second);
                    result.success = true;
                    std::cout << "Using STS x-subject-token as session token" << std::endl;
                } else {
                    result.errorMessage = "Empty STS response body and no x-subject-token header";
                }
            }
        } else {
            result.errorMessage = "STS API returned error code: " + 
                std::to_string(static_cast<int>(stsAwsResult.GetResponseCode()));
        }
        
    } catch (const std::exception& e) {
        result.errorMessage = "Exception during STS call: " + std::string(e.what());
        std::cout << "STS call exception: " << e.what() << std::endl;
    } catch (...) {
        result.errorMessage = "Unknown exception during STS call";
        std::cout << "Unknown STS call exception" << std::endl;
    }
    
    std::cout << "=== Step 2 Complete ===" << std::endl;
    return result;
}

}  // namespace Internal
}  // namespace Aws