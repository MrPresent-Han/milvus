package huawei

import (
	"github.com/cockroachdb/errors"
	"github.com/minio/minio-go/v7"
	minioCred "github.com/minio/minio-go/v7/pkg/credentials"
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
}

func (p *HuaweiCredentialProvider) Retrieve() (minioCred.Value, error) {
	return minioCred.Value{}, nil
}

func (p *HuaweiCredentialProvider) IsExpired() bool {
	return false
}
