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
	fmt.Println("First 10 IDs:", ids[:10])
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

	fmt.Println("-------------------------------")
	// Create a model with one weight per feature-candidate pair and a fixed seed.
	m := NewModel(tk.Size(), 42)

	// Predict the word after the first training example's context, "THE" "TRAGEDY".
	probs := m.Forward(examples[0].Context[0], examples[0].Context[1])

	// The probabilities should add up to about 1.
	var total float32
	for _, p := range probs { // Add up every probability.
		total += p
	}
	fmt.Println("Sum of probabilities:", total) // Should print a value very close to 1.

	// Show how much probability the untrained model gives to the correct answer, "OF".
	fmt.Println("P(correct):", probs[examples[0].Target]) // Around 1/7556 ≈ 0.00013, since weights are random.

	fmt.Println("-------------------------------")
	fmt.Printf("loss before training: %.4f\n", m.AverageLoss(examples))
	m.Train(examples, 10, 0.1)
	fmt.Printf("loss after training:  %.4f\n", m.AverageLoss(examples))

	GradientCheck(m, examples[0], 1e-2)

	fmt.Println("-------------------------------")
	start := [2]int{examples[0].Context[0], examples[0].Context[1]}
	for _, temp := range []float32{0.5, 1.0, 1.5} {
		fmt.Printf("temperature %.1f:\n%s\n\n", temp, Generate(m, tk, start, 40, temp, 1))
	}
}
