/*
 * json_parser.hpp
 * ===============
 * A minimal, self-contained JSON parser sufficient for reading the
 * circuit_params.json file produced by export_and_parse_onnx.py.
 *
 * Supports: objects, arrays, strings, integers, floats, booleans, null.
 * NOT a full JSON spec implementation — just enough for our use case.
 *
 * For production projects, consider nlohmann/json instead.
 */

#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>
#include <cctype>

class JsonValue {
public:
    enum Type { JNULL, JBOOL, JINT, JFLOAT, JSTRING, JARRAY, JOBJECT };

    Type type;
    long long int_val;
    double float_val;
    bool bool_val;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    JsonValue() : type(JNULL), int_val(0), float_val(0), bool_val(false) {}

    int as_int() const {
        if (type == JINT) return (int)int_val;
        if (type == JFLOAT) return (int)float_val;
        throw std::runtime_error("JSON value is not a number");
    }

    long long as_long() const {
        if (type == JINT) return int_val;
        if (type == JFLOAT) return (long long)float_val;
        throw std::runtime_error("JSON value is not a number");
    }

    const std::string& as_string() const {
        if (type != JSTRING) throw std::runtime_error("JSON value is not a string");
        return str_val;
    }

    const std::vector<JsonValue>& as_array() const {
        if (type != JARRAY) throw std::runtime_error("JSON value is not an array");
        return arr_val;
    }

    const JsonValue& operator[](const std::string& key) const {
        if (type != JOBJECT) throw std::runtime_error("JSON value is not an object");
        auto it = obj_val.find(key);
        if (it == obj_val.end()) throw std::runtime_error("Key not found: " + key);
        return it->second;
    }
};


class JsonParser {
    const std::string& src;
    size_t pos;

    void skip_ws() {
        while (pos < src.size() && std::isspace(src[pos])) pos++;
    }

    char peek() { skip_ws(); return pos < src.size() ? src[pos] : '\0'; }
    char next() { skip_ws(); return pos < src.size() ? src[pos++] : '\0'; }

    std::string parse_string_literal() {
        if (next() != '"') throw std::runtime_error("Expected '\"'");
        std::string result;
        while (pos < src.size() && src[pos] != '"') {
            if (src[pos] == '\\') {
                pos++;
                if (pos < src.size()) {
                    switch (src[pos]) {
                        case '"': result += '"'; break;
                        case '\\': result += '\\'; break;
                        case 'n': result += '\n'; break;
                        case 't': result += '\t'; break;
                        case '/': result += '/'; break;
                        default: result += src[pos]; break;
                    }
                }
            } else {
                result += src[pos];
            }
            pos++;
        }
        if (pos < src.size()) pos++; // skip closing "
        return result;
    }

    JsonValue parse_number() {
        skip_ws();
        size_t start = pos;
        bool is_float = false;
        if (src[pos] == '-') pos++;
        while (pos < src.size() && std::isdigit(src[pos])) pos++;
        if (pos < src.size() && src[pos] == '.') { is_float = true; pos++; }
        while (pos < src.size() && std::isdigit(src[pos])) pos++;
        if (pos < src.size() && (src[pos] == 'e' || src[pos] == 'E')) {
            is_float = true;
            pos++;
            if (pos < src.size() && (src[pos] == '+' || src[pos] == '-')) pos++;
            while (pos < src.size() && std::isdigit(src[pos])) pos++;
        }
        std::string num_str = src.substr(start, pos - start);
        JsonValue v;
        if (is_float) {
            v.type = JsonValue::JFLOAT;
            v.float_val = std::stod(num_str);
        } else {
            v.type = JsonValue::JINT;
            v.int_val = std::stoll(num_str);
        }
        return v;
    }

public:
    JsonParser(const std::string& source) : src(source), pos(0) {}

    JsonValue parse() {
        skip_ws();
        char c = peek();

        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::JSTRING;
            v.str_val = parse_string_literal();
            return v;
        }
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(c)) return parse_number();

        throw std::runtime_error(std::string("Unexpected character: ") + c);
    }

    JsonValue parse_object() {
        JsonValue v;
        v.type = JsonValue::JOBJECT;
        next(); // consume '{'
        if (peek() == '}') { next(); return v; }
        while (true) {
            std::string key = parse_string_literal();
            if (next() != ':') throw std::runtime_error("Expected ':'");
            v.obj_val[key] = parse();
            char c = next();
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("Expected ',' or '}'");
        }
        return v;
    }

    JsonValue parse_array() {
        JsonValue v;
        v.type = JsonValue::JARRAY;
        next(); // consume '['
        if (peek() == ']') { next(); return v; }
        while (true) {
            v.arr_val.push_back(parse());
            char c = next();
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("Expected ',' or ']'");
        }
        return v;
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.type = JsonValue::JBOOL;
        if (src.substr(pos, 4) == "true") { v.bool_val = true; pos += 4; }
        else if (src.substr(pos, 5) == "false") { v.bool_val = false; pos += 5; }
        else throw std::runtime_error("Invalid boolean");
        return v;
    }

    JsonValue parse_null() {
        if (src.substr(pos, 4) != "null") throw std::runtime_error("Invalid null");
        pos += 4;
        return JsonValue();
    }
};

inline JsonValue json_parse(const std::string& json_str) {
    JsonParser parser(json_str);
    return parser.parse();
}

#endif // JSON_PARSER_HPP
