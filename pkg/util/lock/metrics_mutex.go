package lock

import (
	"strconv"
	"sync"
	"time"

	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"github.com/timandy/routine"
	"go.uber.org/zap"
)

var enableLockMetrics bool
var logWarnThresHold time.Duration
var logInfoThresHold time.Duration

type MetricsRWMutex struct {
	mutex          sync.RWMutex
	lockName       string
	acquireTimeMap *typeutil.ConcurrentMap[string, time.Time]
}

const (
	readLock  = "READ_LOCK"
	writeLock = "WRITE_LOCK"
	hold      = "HOLD"
	acquire   = "ACQUIRE"
)

func encodeSourceKey(source string) string {
	return source + "-" + strconv.FormatInt(routine.Goid(), 10)
}

func (mRWLock *MetricsRWMutex) RLock(source string) {
	if enableLockMetrics {
		before := time.Now()
		mRWLock.mutex.RLock()
		key := encodeSourceKey(source)
		mRWLock.acquireTimeMap.Insert(key, time.Now())
		logLock(time.Since(before), mRWLock.lockName, key, readLock, acquire)
	} else {
		mRWLock.mutex.RLock()
	}
}

func (mRWLock *MetricsRWMutex) Lock(source string) {
	if enableLockMetrics {
		before := time.Now()
		mRWLock.mutex.Lock()
		key := encodeSourceKey(source)
		mRWLock.acquireTimeMap.Insert(key, time.Now())
		logLock(time.Since(before), mRWLock.lockName, key, writeLock, acquire)
	} else {
		mRWLock.mutex.Lock()
	}
}

func (mRWLock *MetricsRWMutex) UnLock(source string) {
	mRWLock.maybeLogUnlockDuration(source, writeLock)
	mRWLock.mutex.Unlock()
}

func (mRWLock *MetricsRWMutex) RUnLock(source string) {
	mRWLock.maybeLogUnlockDuration(source, readLock)
	mRWLock.mutex.RUnlock()
}

func (mRWLock *MetricsRWMutex) maybeLogUnlockDuration(source string, lockType string) {
	if enableLockMetrics {
		key := encodeSourceKey(source)
		acquireTime, ok := mRWLock.acquireTimeMap.Get(key)
		if ok {
			logLock(time.Since(acquireTime), mRWLock.lockName, key, lockType, hold)
			mRWLock.acquireTimeMap.GetAndRemove(key)
		} else {
			log.Error("there's no lock history for the source, there may be some defects in codes",
				zap.String("source", source))
		}
	}
}

func logLock(duration time.Duration, lockName string, source string, lockType string, opType string) {
	if duration >= logWarnThresHold {
		log.Warn("lock takes too long", zap.String("lockName", lockName), zap.String("lockType", lockType),
			zap.String("source", source), zap.String("opType", opType),
			zap.Duration("time_cost", duration))
	} else if duration >= logInfoThresHold {
		log.Info("lock takes too long", zap.String("lockName", lockName), zap.String("lockType", lockType),
			zap.String("source", source), zap.String("opType", opType),
			zap.Duration("time_cost", duration))
	}
	metrics.LockCosts.WithLabelValues(lockName, source, lockType, opType).Set(float64(duration.Milliseconds()))
}

func NewMetricsLock(name string) *MetricsRWMutex {
	enableLockMetrics = paramtable.Get().CommonCfg.EnableLockMetrics.GetAsBool()
	logWarnThresHold = paramtable.Get().CommonCfg.LockSlowLogWarnThreshold.GetAsDuration(time.Millisecond)
	logInfoThresHold = paramtable.Get().CommonCfg.LockSlowLogInfoThreshold.GetAsDuration(time.Millisecond)
	return &MetricsRWMutex{
		lockName:       name,
		acquireTimeMap: typeutil.NewConcurrentMap[string, time.Time](),
	}
}
