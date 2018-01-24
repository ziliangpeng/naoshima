# Goroutine

- super lightweight execution flow
- language built-in scheduling
- if blocked, language knows to put it aside
- communicate through `channel`
- spits out value to channel
- auto resume after value is consumed
- language can schedule high number of goroutines on optimal number of threads
- programmer think in single flow, no need to coordinate threads
