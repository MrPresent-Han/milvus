// Code generated by mockery v2.32.4. DO NOT EDIT.

package proxy

import (
	context "context"

	grpc "google.golang.org/grpc"

	mock "github.com/stretchr/testify/mock"

	rootcoordpb "github.com/milvus-io/milvus/internal/proto/rootcoordpb"
)

// mockTimestampAllocator is an autogenerated mock type for the timestampAllocatorInterface type
type mockTimestampAllocator struct {
	mock.Mock
}

type mockTimestampAllocator_Expecter struct {
	mock *mock.Mock
}

func (_m *mockTimestampAllocator) EXPECT() *mockTimestampAllocator_Expecter {
	return &mockTimestampAllocator_Expecter{mock: &_m.Mock}
}

// AllocTimestamp provides a mock function with given fields: ctx, req, opts
func (_m *mockTimestampAllocator) AllocTimestamp(ctx context.Context, req *rootcoordpb.AllocTimestampRequest, opts ...grpc.CallOption) (*rootcoordpb.AllocTimestampResponse, error) {
	_va := make([]interface{}, len(opts))
	for _i := range opts {
		_va[_i] = opts[_i]
	}
	var _ca []interface{}
	_ca = append(_ca, ctx, req)
	_ca = append(_ca, _va...)
	ret := _m.Called(_ca...)

	var r0 *rootcoordpb.AllocTimestampResponse
	var r1 error
	if rf, ok := ret.Get(0).(func(context.Context, *rootcoordpb.AllocTimestampRequest, ...grpc.CallOption) (*rootcoordpb.AllocTimestampResponse, error)); ok {
		return rf(ctx, req, opts...)
	}
	if rf, ok := ret.Get(0).(func(context.Context, *rootcoordpb.AllocTimestampRequest, ...grpc.CallOption) *rootcoordpb.AllocTimestampResponse); ok {
		r0 = rf(ctx, req, opts...)
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(*rootcoordpb.AllocTimestampResponse)
		}
	}

	if rf, ok := ret.Get(1).(func(context.Context, *rootcoordpb.AllocTimestampRequest, ...grpc.CallOption) error); ok {
		r1 = rf(ctx, req, opts...)
	} else {
		r1 = ret.Error(1)
	}

	return r0, r1
}

// mockTimestampAllocator_AllocTimestamp_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'AllocTimestamp'
type mockTimestampAllocator_AllocTimestamp_Call struct {
	*mock.Call
}

// AllocTimestamp is a helper method to define mock.On call
//   - ctx context.Context
//   - req *rootcoordpb.AllocTimestampRequest
//   - opts ...grpc.CallOption
func (_e *mockTimestampAllocator_Expecter) AllocTimestamp(ctx interface{}, req interface{}, opts ...interface{}) *mockTimestampAllocator_AllocTimestamp_Call {
	return &mockTimestampAllocator_AllocTimestamp_Call{Call: _e.mock.On("AllocTimestamp",
		append([]interface{}{ctx, req}, opts...)...)}
}

func (_c *mockTimestampAllocator_AllocTimestamp_Call) Run(run func(ctx context.Context, req *rootcoordpb.AllocTimestampRequest, opts ...grpc.CallOption)) *mockTimestampAllocator_AllocTimestamp_Call {
	_c.Call.Run(func(args mock.Arguments) {
		variadicArgs := make([]grpc.CallOption, len(args)-2)
		for i, a := range args[2:] {
			if a != nil {
				variadicArgs[i] = a.(grpc.CallOption)
			}
		}
		run(args[0].(context.Context), args[1].(*rootcoordpb.AllocTimestampRequest), variadicArgs...)
	})
	return _c
}

func (_c *mockTimestampAllocator_AllocTimestamp_Call) Return(_a0 *rootcoordpb.AllocTimestampResponse, _a1 error) *mockTimestampAllocator_AllocTimestamp_Call {
	_c.Call.Return(_a0, _a1)
	return _c
}

func (_c *mockTimestampAllocator_AllocTimestamp_Call) RunAndReturn(run func(context.Context, *rootcoordpb.AllocTimestampRequest, ...grpc.CallOption) (*rootcoordpb.AllocTimestampResponse, error)) *mockTimestampAllocator_AllocTimestamp_Call {
	_c.Call.Return(run)
	return _c
}

// newMockTimestampAllocator creates a new instance of mockTimestampAllocator. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func newMockTimestampAllocator(t interface {
	mock.TestingT
	Cleanup(func())
}) *mockTimestampAllocator {
	mock := &mockTimestampAllocator{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}