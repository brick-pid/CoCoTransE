main() {
    try{
    let correct = ArrayList<Bool>(
            solve(1000) == "1",
            solve(150) == "110",
            solve(147) == "1100",
            solve(333) == "1001",
            solve(963) == "10010"
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