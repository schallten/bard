package main

import (
	"bufio"
	"os"
	"strings"
)

type Tokenizer struct {
	toID       map[string]int
	tokens     []string
	occurences int
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{toID: make(map[string]int)}
}

// records one occurence of a token and returns its ID
func (t *Tokenizer) Add(tok string) int {
	t.occurences++
	// token already exists
	if id, ok := t.toID[tok]; ok {
		return id
	}
	id := len(t.tokens) // next free id
	t.toID[tok] = id
	t.tokens = append(t.tokens, tok)
	return id
}

func (t *Tokenizer) Encode(tok string) (int, bool) {
	id, ok := t.toID[tok]
	return id, ok
}

// id back to token
func (t *Tokenizer) Decode(id int)
