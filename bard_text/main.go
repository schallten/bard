package main

import (
	"fmt"
	"log"
)

func main() {
	tk, err := LoadFile("hamlet_tokens.txt")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("occurences: ", tk.Occurences())
	fmt.Println("Vocabulary size:", tk.Size())

	for id := 0; id < tk.Size(); id++ {
		tok, ok := tk.Decode(id) // id to token
		if !ok {
			log.Fatalf("decode failed for id %d", id)
		}
		back, ok := tk.Encode(tok) // token to id
		if !ok || back != id {
			log.Fatalf("round trip failed for id %d (%q)", id, tok)
		}
	}
	ids, err := EncodeFile("hamlet_tokens.txt", tk)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Encoded length:", len(ids)) // Should be 31502.
	fmt.Println("First 10 IDs:", ids[:10])   // Check these against your file by hand.
	fmt.Println("ok")

}
