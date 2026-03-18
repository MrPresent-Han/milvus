package huawei

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/minio/minio-go/v7"
	minioCred "github.com/minio/minio-go/v7/pkg/credentials"
)

// TestSTSTokenConflict verifies whether issuing a second STS token
// invalidates the first one on HuaweiCloud IAM.
//
// Run manually on a cloud instance with proper env vars set:
//   HUAWEICLOUD_SDK_AK, HUAWEICLOUD_SDK_SK, HUAWEICLOUD_SDK_REGION
//   MILVUS_OBS_BUCKET (the OBS bucket to test against)
//   MILVUS_OBS_ENDPOINT (e.g. "obs.cn-east-3.myhuaweicloud.com:443")
//
// go test -v -run TestSTSTokenConflict -tags dynamic ./pkg/objectstorage/huawei/
func TestSTSTokenConflict(t *testing.T) {
	// Skip in CI — this is a manual integration test.
	// The Go SDK's BasicCredentialProviderChain uses OIDC when idpId+idTokenFile are set.
	region := os.Getenv("HUAWEICLOUD_SDK_REGION")
	idTokenFile := os.Getenv("HUAWEICLOUD_SDK_ID_TOKEN_FILE")
	if region == "" || idTokenFile == "" {
		t.Skip("skipping: HUAWEICLOUD_SDK_REGION or HUAWEICLOUD_SDK_ID_TOKEN_FILE not set")
	}

	bucket := os.Getenv("MILVUS_OBS_BUCKET")
	if bucket == "" {
		bucket = "zilliz-prod-hvjibd0vkaxr1p" // default for HuaweiCloud test instances
	}
	endpoint := os.Getenv("MILVUS_OBS_ENDPOINT")
	if endpoint == "" {
		endpoint = fmt.Sprintf("obs.%s.myhuaweicloud.com:443", region)
	}

	// Reset singleton to ensure clean state
	globalCredProviderMu.Lock()
	globalCredProvider = nil
	globalCredProviderMu.Unlock()

	provider := NewCredentialProvider().(*HuaweiCredentialProvider)

	// --- Get Token 1 ---
	creds1, err := provider.Retrieve()
	if err != nil {
		t.Fatalf("failed to retrieve token1: %v", err)
	}
	t.Logf("Token1: AK=%s..., expiration=%v", creds1.AccessKeyID[:8], creds1.Expiration)

	// Verify token1 works
	err = obsListObjects(endpoint, bucket, creds1)
	if err != nil {
		t.Fatalf("token1 failed immediately after issuance: %v", err)
	}
	t.Log("Token1 works immediately after issuance ✓")

	// --- Force a second STS call by resetting cached state ---
	provider.refreshMu.Lock()
	provider.expiration = time.Time{} // zero value forces STS call
	provider.refreshMu.Unlock()

	time.Sleep(2 * time.Second) // small gap

	// --- Get Token 2 ---
	creds2, err := provider.Retrieve()
	if err != nil {
		t.Fatalf("failed to retrieve token2: %v", err)
	}
	t.Logf("Token2: AK=%s..., expiration=%v", creds2.AccessKeyID[:8], creds2.Expiration)

	if creds1.AccessKeyID == creds2.AccessKeyID {
		t.Log("WARNING: token1 and token2 have the same AK — IAM may have returned cached token")
	} else {
		t.Log("Token1 and Token2 have different AKs ✓")
	}

	// Verify token2 works
	err = obsListObjects(endpoint, bucket, creds2)
	if err != nil {
		t.Fatalf("token2 failed: %v", err)
	}
	t.Log("Token2 works ✓")

	// --- THE KEY TEST: does token1 still work after token2 was issued? ---
	err = obsListObjects(endpoint, bucket, creds1)
	if err != nil {
		t.Errorf("TOKEN CONFLICT DETECTED: token1 is INVALID after token2 was issued: %v", err)
		t.Log("CONCLUSION: HuaweiCloud IAM invalidates previous STS tokens when new ones are issued")
	} else {
		t.Log("Token1 still works after Token2 was issued ✓")
		t.Log("CONCLUSION: STS tokens coexist — no mutual invalidation")
	}

	// Cleanup
	globalCredProviderMu.Lock()
	globalCredProvider = nil
	globalCredProviderMu.Unlock()
}

func obsListObjects(endpoint, bucket string, creds minioCred.Value) error {
	client, err := minio.New(endpoint, &minio.Options{
		Creds:  minioCred.NewStaticV4(creds.AccessKeyID, creds.SecretAccessKey, creds.SessionToken),
		Secure: true,
		Region: "cn-east-3",
		BucketLookup: minio.BucketLookupDNS,
	})
	if err != nil {
		return fmt.Errorf("create minio client: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Just list 1 object to verify the token works
	objCh := client.ListObjects(ctx, bucket, minio.ListObjectsOptions{
		MaxKeys: 1,
	})
	for obj := range objCh {
		if obj.Err != nil {
			return fmt.Errorf("list objects: %w", obj.Err)
		}
	}
	return nil
}
