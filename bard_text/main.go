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

	examples := BuildExamples(ids)
	fmt.Println("Examples:", len(examples)) // Should be 31500.

	// Confirm every ID is in range, since the largest should be 7555.
	maxID := 0
	for _, id := range ids {
		if id > maxID {
			maxID = id
		}
	}
	fmt.Println("Max ID:", maxID) // Should be 7555.

	// Print the first few examples as words to check alignment.
	PrintExamples(tk, examples, 5)

}
