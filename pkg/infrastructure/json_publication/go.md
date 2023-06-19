SessionWithContext represents a logical connection (which is not tied to a physical connection) to the server

```go
https://neo4j.com/docs/go-manual/current/client-applications/
func createDriver(uri, username, password string) (neo4j.DriverWithContext, error) {
	return neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
}
```

// call on application exit
func closeDriver(ctx context.Context, driver neo4j.DriverWithContext) error {
	return driver.Close(ctx)
}