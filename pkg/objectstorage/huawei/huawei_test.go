package huawei

import (
	"fmt"
	"testing"

	"github.com/huaweicloud/huaweicloud-sdk-go-v3/core/auth/provider"
	"github.com/huaweicloud/huaweicloud-sdk-go-v3/core/config"
	iam "github.com/huaweicloud/huaweicloud-sdk-go-v3/services/iam/v3"
	model "github.com/huaweicloud/huaweicloud-sdk-go-v3/services/iam/v3/model"
	iamRegion "github.com/huaweicloud/huaweicloud-sdk-go-v3/services/iam/v3/region"
)

func TestNewMinioClient(t *testing.T) {
	// providers := []provider.ICredentialProvider{
	// 	provider.BasicCredentialEnvProvider(),
	// 	provider.BasicCredentialProfileProvider(),
	// 	provider.BasicCredentialMetadataProvider(),
	// }
	// chain := provider.NewCredentialProviderChain(providers)
	// cred, _ := chain.GetCredentials()
	// fmt.Println(cred.GetSecretId())
	basicChain := provider.BasicCredentialProviderChain()
	basicCred, err := basicChain.GetCredentials()
	if err != nil {
		t.Fatalf("failed to get credentials: %v", err)
	}

	hcClient, err := iam.IamClientBuilder().
		WithRegion(iamRegion.CN_EAST_3).
		WithCredential(basicCred).
		WithHttpConfig(config.DefaultHttpConfig()).
		SafeBuild()
	if err != nil {
		fmt.Println(err)
		return
	}
	client := iam.NewIamClient(hcClient)

	request := &model.CreateTemporaryAccessKeyByTokenRequest{
		Body: &model.CreateTemporaryAccessKeyByTokenRequestBody{
			Auth: &model.TokenAuth{
				Identity: &model.TokenAuthIdentity{
					Methods: []model.TokenAuthIdentityMethods{model.GetTokenAuthIdentityMethodsEnum().TOKEN},
				},
			},
		},
	}
	response, err := client.CreateTemporaryAccessKeyByToken(request)
	if err != nil {
		t.Fatalf("failed to create temporary access key: %v", err)
	}
	fmt.Println("response.Credential-ak: ", response.Credential.Access)
	fmt.Println("response.Credential-sk: ", response.Credential.Secret)
	fmt.Println("response.Credential-securitytoken: ", response.Credential.Securitytoken)
	fmt.Println("response.Credential-expires_at: ", response.Credential.ExpiresAt)
}

// {
// 	// 读取OIDC token
// 	tokenData, _ := ioutil.ReadFile("/var/run/secrets/tokens/oidc-token")
// 	userToken := strings.TrimSpace(string(tokenData))
// 	endPoints := []string{"https://iam.cn-east-3.myhuaweicloud.com/v3.0/OS-AUTH/id-token/tokens"}

// 	httpConfig := config.DefaultHttpConfig()
// 	httpConfig.UserAgent = utils.GetEnvInfoString()
// 	defaultHttpClient := impl.NewDefaultHttpClient(httpConfig)
// 	hcHttpClient := httpclient.NewHcHttpClient(defaultHttpClient).WithEndpoints(endPoints).
// 		WithCredential(credentials).WithErrorHandler(builder.errorHandler)

// 	// 创建IAM客户端

// 	iamClient := v3.NewIamClient(hcClient)

// 	// 构建CreateTemporaryAccessKeyByToken请求
// 	requestBody := &model.CreateTemporaryAccessKeyByTokenRequestBody{
// 		Auth: &model.TokenAuth{
// 			Identity: &model.TokenAuthIdentity{
// 				Methods: []model.TokenAuthIdentityMethods{
// 					model.GetTokenAuthIdentityMethodsEnum().TOKEN,
// 				},
// 			},
// 		},
// 	}

// 	request := &model.CreateTemporaryAccessKeyByTokenRequest{
// 		Body: requestBody,
// 	}

// 	// 设置X-Auth-Token头
// 	hcClient.WithHttpConfig(config.DefaultHttpConfig().
// 		WithIgnoreSSLVerification(true).
// 		WithHttpHandler(func(httpRequest *httpclient.HttpRequest) {
// 			httpRequest.AddHeaderParam("X-Auth-Token", userToken)
// 		}))

// 	// 调用CreateTemporaryAccessKeyByToken获取临时凭证
// 	response, err := iamClient.CreateTemporaryAccessKeyByToken(request)
// 	if err != nil {
// 		t.Fatalf("failed to create temporary access key: %v", err)
// 	}

// 	// 打印获得的AK, SK, Security Token
// 	if response.Credential != nil {
// 		cred := response.Credential
// 		fmt.Printf("=== 华为云临时凭证 ===\n")
// 		fmt.Printf("AK: %s\n", cred.Access)
// 		fmt.Printf("SK: %s\n", cred.Secret)
// 		fmt.Printf("Security Token: %s\n", cred.Securitytoken)
// 		fmt.Printf("Expires At: %s\n", cred.ExpiresAt)
// 		fmt.Printf("=====================\n")

// 		t.Logf("Successfully obtained temporary credentials")
// 		t.Logf("AK: %s", cred.Access)
// 		t.Logf("SK: %s", cred.Secret)
// 		t.Logf("Security Token: %s", cred.Securitytoken)
// 		t.Logf("Expires At: %s", cred.ExpiresAt)
// 	} else {
// 		t.Fatalf("No credential returned")
// 	}
// }
