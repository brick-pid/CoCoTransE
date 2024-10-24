main() {
    try{
        let correct = ArrayList<Bool>(
                filterIntergers(new ArrayList<>(Arrays.asList(4, Map.of(), List.of(), 23.2, 9, "adasd"))).equals(Arrays.asList(4, 9)),
                filterIntergers(new ArrayList<>(Arrays.asList(3, 'c', 3, 3, 'a', 'b'))).equals(Arrays.asList(3, 3, 3))
        );
        if (correct.contains(false)) {
                    print(correct)
            }
        }
    catch(e: Exception)
    {
        print("false")
    }
}