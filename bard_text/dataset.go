package main

import "fmt"

type Example struct {
	// one trainign case : two context ids and id that should come next
	Context [2]int // two prev tokens
	Target  int    // id of word taht actually forllows them
}

// slides a window of three IDs across sequence
// each window has one examp,e , first two ids are context and third is target
func BuildExamples(ids []int) []Example {
	if len(ids) < 3 {
		return nil
	}
	examples := make([]Example, 0, len(ids)-2)
	for i := 0; i+2 < len(ids); i++ {
		ex := Example{
			Context: [2]int{ids[i], ids[i+1]},
			Target:  ids[i+2],
		}
		examples = append(examples, ex) // add example to result
	}
	return examples
}

func PrintExamples(tk *Tokenizer, examples []Example, n int) {
	if n > len(examples) {
		n = len(examples)
	}
	for _, ex := range examples[:n] {
		a, _ := tk.Decode(ex.Context[0])
		b, _ := tk.Decode(ex.Context[1])
		c, _ := tk.Decode(ex.Target)
		fmt.Printf("%q %q -> %q\n", a, b, c)
	}
}
