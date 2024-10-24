main() {
    try{
    let correct = ArrayList<Bool>(
            parseNestedParens("(()()) ((())) () ((())()())") == (ArrayList<Int64>(2, 3, 1, 3)),
            parseNestedParens("() (()) ((())) (((())))") == (ArrayList<Int64>(1, 2, 3, 4)),
            parseNestedParens("(()(())((())))") == (ArrayList<Int64>([4]))
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