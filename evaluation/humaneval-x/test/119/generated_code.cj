main() {
    try{
    let correct = ArrayList<Bool>(
            matchParens(ArrayList<String>("()(", ")"))=="Yes",
            matchParens(ArrayList<String>(")", ")"))=="No",
            matchParens(ArrayList<String>("(()(())", "())())"))=="No",
            matchParens(ArrayList<String>(")())", "(()()("))=="Yes",
            matchParens(ArrayList<String>("(())))", "(()())(("))=="Yes",
            matchParens(ArrayList<String>("()", "())"))=="No",
            matchParens(ArrayList<String>("(()(", "()))()"))=="Yes",
            matchParens(ArrayList<String>("((((", "((())"))=="No",
            matchParens(ArrayList<String>(")(()", "(()("))=="No",
            matchParens(ArrayList<String>(")(", ")("))=="No"
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