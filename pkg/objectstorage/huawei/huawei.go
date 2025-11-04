package huawei

import (
	"os"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/minio/minio-go/v7"
	minioCred "github.com/minio/minio-go/v7/pkg/credentials"

	"github.com/huaweicloud/huaweicloud-sdk-go-v3/core/auth"
	"github.com/huaweicloud/huaweicloud-sdk-go-v3/core/auth/provider"
	"github.com/huaweicloud/huaweicloud-sdk-go-v3/core/config"
	"github.com/huaweicloud/huaweicloud-sdk-go-v3/core/region"
	iam "github.com/huaweicloud/huaweicloud-sdk-go-v3/services/iam/v3"
	"github.com/huaweicloud/huaweicloud-sdk-go-v3/services/iam/v3/model"
	iamRegion "github.com/huaweicloud/huaweicloud-sdk-go-v3/services/iam/v3/region"
)

func NewMinioClient(address string, opts *minio.Options) (*minio.Client, error) {
	if opts == nil {
		opts = &minio.Options{}
	}
	if opts.Creds == nil {
		credProvider, err := NewCredentialProvider()
		if err != nil {
			return nil, errors.Wrap(err, "failed to create credential provider")
		}
		opts.Creds = minioCred.New(credProvider)
	}
	if address == "" {
		address = "obs.myhuaweicloud.com"
		opts.Secure = true
	}
	return minio.New(address, opts)
}

func NewCredentialProvider() (minioCred.Provider, error) {
	return &HuaweiCredentialProvider{}, nil
}

type HuaweiCredentialProvider struct {
	credentials minioCred.Value
	expiration  time.Time

	// 缓存的客户端组件
	basicCred auth.ICredential
	regionObj *region.Region
	iamClient *iam.IamClient

	// 初始化锁
	initOnce sync.Once
	initErr  error
}

// 初始化客户端组件（只执行一次）
func (p *HuaweiCredentialProvider) initClients() {
	p.initOnce.Do(func() {
		// 使用华为云SDK获取基础凭证
		basicChain := provider.BasicCredentialProviderChain()
		basicCred, err := basicChain.GetCredentials()
		if err != nil {
			p.initErr = errors.Wrap(err, "failed to get basic credentials")
			return
		}
		p.basicCred = basicCred

		// 从环境变量读取region
		regionName := os.Getenv("HUAWEICLOUD_SDK_REGION")
		if regionName == "" {
			regionName = "cn-east-3" // 默认region
		}

		// 根据region名称获取对应的region对象
		regionObj, err := iamRegion.SafeValueOf(regionName)
		if err != nil {
			// 如果region不支持，使用默认region
			regionObj, _ = iamRegion.SafeValueOf("cn-east-3")
		}
		p.regionObj = regionObj

		// 创建IAM客户端
		hcClient, err := iam.IamClientBuilder().
			WithRegion(p.regionObj).
			WithCredential(p.basicCred).
			WithHttpConfig(config.DefaultHttpConfig()).
			SafeBuild()
		if err != nil {
			p.initErr = errors.Wrap(err, "failed to build IAM client")
			return
		}
		p.iamClient = iam.NewIamClient(hcClient)
	})
}

func (p *HuaweiCredentialProvider) Retrieve() (minioCred.Value, error) {
	// 确保客户端已初始化
	p.initClients()
	if p.initErr != nil {
		return minioCred.Value{}, p.initErr
	}

	// 构建CreateTemporaryAccessKeyByToken请求
	request := &model.CreateTemporaryAccessKeyByTokenRequest{
		Body: &model.CreateTemporaryAccessKeyByTokenRequestBody{
			Auth: &model.TokenAuth{
				Identity: &model.TokenAuthIdentity{
					Methods: []model.TokenAuthIdentityMethods{model.GetTokenAuthIdentityMethodsEnum().TOKEN},
				},
			},
		},
	}

	// 调用API获取临时凭证
	response, err := p.iamClient.CreateTemporaryAccessKeyByToken(request)
	if err != nil {
		return minioCred.Value{}, errors.Wrap(err, "failed to create temporary access key")
	}

	if response.Credential == nil {
		return minioCred.Value{}, errors.New("no credential returned from Huawei Cloud")
	}

	// 解析过期时间
	expiration, err := time.Parse("2006-01-02T15:04:05Z", response.Credential.ExpiresAt)
	if err != nil {
		return minioCred.Value{}, errors.Wrap(err, "failed to parse expiration time")
	}

	// 构建minio凭证
	credentials := minioCred.Value{
		AccessKeyID:     response.Credential.Access,
		SecretAccessKey: response.Credential.Secret,
		SessionToken:    response.Credential.Securitytoken,
		SignerType:      minioCred.SignatureV4,
	}

	// 缓存凭证和过期时间
	p.credentials = credentials
	p.expiration = expiration

	return credentials, nil
}

func (p *HuaweiCredentialProvider) IsExpired() bool {
	// 提前5分钟刷新凭证
	return time.Now().UTC().After(p.expiration.Add(-5 * time.Minute))
}
