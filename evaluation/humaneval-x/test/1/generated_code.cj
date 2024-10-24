main() {
    try{
        let correct = ArrayList<Bool>(
                separateParenGroups("(()()) ((())) () ((())()())") == (ArrayList<String>(
                        "(()())", "((()))", "()", "((())()())"
                )),
                separateParenGroups("() (()) ((())) (((())))") == (ArrayList<String>(
                        "()", "(())", "((()))", "(((())))"
                )),
                separateParenGroups("(()(())((())))") == (ArrayList<String>(
                        "(()(())((())))"
                )),
                separateParenGroups("( ) (( )) (( )( ))") == (ArrayList<String>("()", "(())", "(()())"))
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