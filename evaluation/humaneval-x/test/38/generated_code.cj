var letters = "abcdefghijklmnopqrstuvwxyz".toArray();
let rand = Random(42);
public func random_string(length : Int64): String {
    var sb = StringBuilder();
    for (i in 0..length) {
        sb.append(letters[rand.nextInt64(26)]);
    }
    return sb.toString();
}
func encodeCyclic(s: String): String {
    var groups = ArrayList<String>()
    var i = 0

    while (i < s.size) {
        groups.append(s[i .. min(i + 3, s.size)])
        i += 3
    }

    i = 0
    while (i < groups.size) {
        if (groups[i].size == 3) {
            groups[i] = groups[i][1..groups[i].size] + groups[i][0..1]
        }
        i += 1
    }

    var result = ""
    for (group in groups) {
        result += group
    }

    result
}
main() {
    try{
        for (i in 0..100) {
            var str = random_string(rand.nextInt64(10) + 10);
            var encode_str = encodeCyclic(str);
            if (!(decodeCyclic(encode_str)==(str))) {
                print("false")
            }
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}