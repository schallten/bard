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
func (t *Tokenizer) Decode(id int) (string, bool) {
	if id < 0 || id >= len(t.tokens) {
		return "", false
	}
	return t.tokens[id], true
}

func (t *Tokenizer) Size() int {
	return len(t.tokens)
} // distinc tokens

func (t *Tokenizer) Occurences() int {
	return t.occurences
} // tokens , repeated included

func LoadFile(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	tk := NewTokenizer() // create empty tokenizer
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		tk.Add(line)
	}
	return tk, sc.Err()
}
