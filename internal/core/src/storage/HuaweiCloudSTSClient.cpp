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

    //m_endpoint = "https://iam.{region}.myhuaweicloud.com/v3/auth/tokens";
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
    
    // 构建请求JSON，使用华为云IAM OIDC格式
    // 根据华为云文档，OIDC认证需要使用mapped方式
    // ss << R"({
    // "auth": {
    //     "identity": {
    //         "methods": ["mapped"],
    //         "mapped": {
    //             "identity_provider": ")" << request.providerId << R"(",
    //             "protocol": "oidc",
    //             "id_token": ")" << request.webIdentityToken << R"("
    //         }
    //     },
    //     "scope": {
    //         "project": {
    //             "id": ")" << request.roleArn << R"("
    //         }
    //     }
    // }
    // })";
    ss << R"({
        "auth": {
          "id_token": {
            "id": ")" << request.webIdentityToken << R"("
          },
          "scope": {
            "project": {
              "id": ")" << request.roleArn << R"("
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
    
    // 正确计算内容长度
    body->seekg(0, body->end);
    auto streamSize = body->tellg();
    body->seekg(0, body->beg);
    
    std::cout << "Request Body Size: " << streamSize << std::endl;
    httpRequest->SetContentLength(std::to_string(streamSize));
    httpRequest->AddContentBody(body);
    httpRequest->SetContentType("application/json; charset=utf-8");

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
    auto responseCode = awsResult.GetResponseCode();
    if (responseCode != Aws::Http::HttpResponseCode::NO_RESPONSE) {
        std::cout << "Response Code: " << static_cast<int>(responseCode) << std::endl;
        
        // 如果是4xx或5xx错误，尝试从错误流中读取响应体
        if (static_cast<int>(responseCode) >= 400) {
            std::cout << "Error response detected, trying to read error stream..." << std::endl;
            
            // 对于400错误，响应体信息通常在GetPayload()中
            std::cout << "Will check payload for error details..." << std::endl;
        }
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
    STSAssumeRoleWithWebIdentityResult result;
    if (subjectTokenIter != responseHeaders.end()) {
        std::cout << "Found x-subject-token in response headers!" << std::endl;
        std::cout << "x-subject-token length: " << subjectTokenIter->second.length() << std::endl;
        const Aws::String subjectToken = subjectTokenIter->second;
        auto stsResult = callHuaweiCloudSTS(subjectToken, request); 
        if (stsResult.success) {
            result.creds = stsResult.credentials;
        }
    } else {
        std::cout << "No x-subject-token in response headers!!!!!!!!" << std::endl;
        return result;
    }

    return result;
}

// HuaweiCloudSTSCredentialsClient::STSCallResult
// HuaweiCloudSTSCredentialsClient::callHuaweiCloudSTS(
//     const Aws::String& userToken, 
//     const STSAssumeRoleWithWebIdentityRequest& request) {
    
//     STSCallResult result;
//     result.success = false;
    
//     std::cout << "=== Step 2: Calling Huawei Cloud STS API ===" << std::endl;
    
//     // 华为云STS API端点
//     //Aws::String stsEndpoint = "https://sts." + request.region + ".myhuaweicloud.com";
//     Aws::String stsEndpoint = "https://iam." + request.region + ".myhuaweicloud.com/v3.0/OS-CREDENTIAL/securitytokens";
//     std::cout << "STS Endpoint: " << stsEndpoint << std::endl;
    
//     // 构建STS请求体 - AssumeRoleWithWebIdentity
    // Aws::StringStream stsRequestBody;
    // stsRequestBody << R"({
    //     "auth": {
    //     "identity": {
    //         "methods": ["token"]
    //     }
    //     }
    // })";

//     std::cout << "STS Request Body: " << stsRequestBody.str() << std::endl;
//     auto respFactory = []() -> Aws::IOStream* {
//         std::cout << "hc===respFactory11111" << std::endl;
//         // 交给 SDK 管理的流：不要在外部保存/释放
//         return Aws::New<StringStream>("STS_RESPONSE");
//     };
//     std::shared_ptr<Aws::Http::HttpRequest> stsHttpRequest(
//         Aws::Http::CreateHttpRequest(
//             stsEndpoint,
//             Aws::Http::HttpMethod::HTTP_POST,
//             respFactory));
    
//     stsHttpRequest->SetUserAgent(Aws::Client::ComputeUserAgentString());
//     stsHttpRequest->SetHeaderValue("Content-Type", "application/json;charset=utf8");
//     stsHttpRequest->SetHeaderValue("X-Auth-Token", userToken);
//     stsHttpRequest->SetHeaderValue("Accept", "application/json");
    
//     // 构建请求体
//     std::shared_ptr<Aws::IOStream> stsBody =
//         Aws::MakeShared<Aws::StringStream>("STS_RESOURCE_CLIENT_LOG_TAG");
//     *stsBody << stsRequestBody.str();
    
//     stsHttpRequest->AddContentBody(stsBody);
//     stsBody->seekg(0, stsBody->end);
//     auto stsStreamSize = stsBody->tellg();
//     stsBody->seekg(0, stsBody->beg);
//     Aws::StringStream stsContentLength;
//     stsContentLength << stsStreamSize;
//     stsHttpRequest->SetContentLength(stsContentLength.str());
    
//     std::cout << "Sending STS request..." << std::endl;
    
//     try {
//         // 发送STS请求
//         auto credentialsStr = GetResourceWithAWSWebServiceResult(stsHttpRequest).GetPayload();
//         std::cout << "STS Response Body: " << credentialsStr << std::endl;
//         std::cout << "STS Response Body Length: " << credentialsStr.length() << std::endl;
//     } catch (...) {
//          result.errorMessage = "Unknown exception during STS call";
//          std::cout << "Unknown STS call exception" << std::endl;
//     }
    
//     std::cout << "=== Step 2 Complete ===" << std::endl;
//     return result;
// }


HuaweiCloudSTSCredentialsClient::STSCallResult
HuaweiCloudSTSCredentialsClient::callHuaweiCloudSTS(
    const Aws::String& userToken, 
    const STSAssumeRoleWithWebIdentityRequest& request) {
        auto httpClient = Http::CreateHttpClient(Client::ClientConfiguration{});

        // 2) 每次返回一个“全新”响应流实例（SDK拥有）
        auto respFactory = []() -> IOStream* {
            // 交给 SDK 管理的流：不要在外部保存/释放
            return Aws::New<StringStream>("STS_RESPONSE");
        };
        
        // 3) 构造 POST 请求（你第二步 /v3.0/OS-CREDENTIAL/securitytokens）
        Aws::String stsEndpoint = "https://iam." + request.region + ".myhuaweicloud.com/v3.0/OS-CREDENTIAL/securitytokens";
        std::cout << "hc===STS Endpoint: " << stsEndpoint << std::endl;
        auto req = Aws::Http::CreateHttpRequest(
            stsEndpoint,
            Http::HttpMethod::HTTP_POST,
            respFactory);
        
        req->SetHeaderValue("Content-Type", "application/json;charset=utf8");
        req->SetHeaderValue("Accept", "application/json");
        req->SetHeaderValue("X-Auth-Token", userToken);
        

        auto body = Aws::MakeShared<StringStream>("STS_REQUEST");
        //*body << R"({"auth":{"identity":{"methods":["assume_role"],"assume_role":{"agency_name":"...","domain_name":"...","duration_seconds":3600}}}})";
        *body << R"({
            "auth": {
            "identity": {
                "methods": ["token"]
            }
            }
        })";
        
        // 正确计算内容长度
        body->seekg(0, body->end);
        auto streamSize = body->tellg();
        body->seekg(0, body->beg);
        
        std::cout << "hc===body stream size: " << streamSize << std::endl;
        req->SetContentLength(std::to_string(streamSize));
        req->AddContentBody(body);


        auto resp = httpClient->MakeRequest(req);
        std::ostringstream oss;
        oss << resp->GetResponseBody().rdbuf();
        Aws::String credentialsStr = oss.str();
        
        std::cout << "hc===STS Response Body: " << credentialsStr << std::endl;
        std::cout << "hc===STS Response Body Length: " << credentialsStr.length() << std::endl;
        
        // 解析华为云STS响应，模仿腾讯云的处理逻辑
        STSCallResult result;
        if (credentialsStr.empty()) {
            result.errorMessage = "Get an empty credential from Huawei Cloud STS";
            std::cout << "hc===Error: " << result.errorMessage << std::endl;
            return result;
        }

        auto json = Utils::Json::JsonView(credentialsStr);
        auto rootNode = json.GetObject("credential");
        if (rootNode.IsNull()) {
            result.errorMessage = "Get credential from STS result failed";
            std::cout << "hc===Error: " << result.errorMessage << std::endl;
            return result;
        }

        // 华为云STS返回的凭证字段名称
        result.credentials.SetAWSAccessKeyId(rootNode.GetString("access"));
        result.credentials.SetAWSSecretKey(rootNode.GetString("secret"));
        result.credentials.SetSessionToken(rootNode.GetString("securitytoken"));
        
        // 解析过期时间
        auto expiresAt = rootNode.GetString("expires_at");
        if (!expiresAt.empty()) {
            result.credentials.SetExpiration(Aws::Utils::DateTime(
                Aws::Utils::StringUtils::Trim(expiresAt.c_str()).c_str(),
                Aws::Utils::DateFormat::ISO_8601));
        }
        result.success = true;
        std::cout << "hc===Successfully parsed credentials from STS response" << std::endl;
        return result;
}

}  // namespace Internal
}  // namespace Aws