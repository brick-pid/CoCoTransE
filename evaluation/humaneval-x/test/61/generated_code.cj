main() {
    try{
    let correct = ArrayList<Bool>(
            correctBracketing("()"),
            correctBracketing("(()())"),
            correctBracketing("()()(()())()"),
            correctBracketing("()()((()()())())(()()(()))"),
            !correctBracketing("((()())))"),
            !correctBracketing(")(()"),
            !correctBracketing("("),
            !correctBracketing("(((("),
            !correctBracketing(")"),
            !correctBracketing("(()"),
            !correctBracketing("()()(()())())(()"),
            !correctBracketing("()()(()())()))()")
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