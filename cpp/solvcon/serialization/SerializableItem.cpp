/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/serialization/SerializableItem.hpp>

namespace solvcon
{

namespace detail
{

inline bool is_json_number(char c)
{
    return std::isdigit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-';
}

inline void throw_serialization_error(std::string const & message, int line, int column)
{
    throw std::runtime_error(std::format("{} (line: {}, column: {})", message, line, column));
}

std::string escape_string(std::string_view str_view)
{
    std::ostringstream oss;
    for (const char c : str_view)
    {
        switch (c)
        {
        case '"':
            oss << "\\\"";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            if (c < 32 || c >= 127)
            {
                oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
            }
            else
            {
                oss << c;
            }
        }
    }
    return oss.str();
}

std::string trim_string(const std::string & str)
{
    const char * whitespace = " \t\n\r\f\v";
    const size_t start = str.find_first_not_of(whitespace);
    const size_t end = str.find_last_not_of(whitespace);
    return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
}

#define SC_DECL_CACULATE_LINE_COLUMN(CH) \
    if (CH == '\n')                      \
    {                                    \
        line += 1;                       \
        column = 1;                      \
    }                                    \
    else                                 \
    {                                    \
        column += 1;                     \
    }

// Scan a balanced {...} or [...] starting at json[index], which must be the
// opening character. Appends the whole balanced expression to out and leaves
// index on the matching closing character. Characters inside quoted strings
// (with backslash escapes) are ignored when matching, so brackets or braces
// embedded in string values do not unbalance the scan. Throws when the
// container is not closed.
inline void scan_balanced_expression(const std::string & json, size_t & index, char open_char, char close_char, int & line, int & column, std::string & out)
{
    int depth = 0;
    bool in_string = false;
    bool escape = false;

    while (index < json.size())
    {
        const char c_curr = json[index];
        out.push_back(c_curr);

        if (escape)
        {
            escape = false;
        }
        else if (in_string)
        {
            if (c_curr == '\\')
            {
                escape = true;
            }
            else if (c_curr == '"')
            {
                in_string = false;
            }
        }
        else if (c_curr == '"')
        {
            in_string = true;
        }
        else if (c_curr == open_char)
        {
            depth += 1;
        }
        else if (c_curr == close_char)
        {
            depth -= 1;
        }

        if (!in_string && depth == 0)
        {
            break;
        }

        SC_DECL_CACULATE_LINE_COLUMN(json[index])
        index += 1;
    }

    if (depth != 0)
    {
        throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
    }
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
// NOLINTNEXTLINE(misc-no-recursion)
JsonArray JsonNode::parse_array(const std::string & json)
{
    JsonArray json_array;
    JsonState state = JsonState::Start;
    std::string value_expression;
    bool after_comma = false;

    int line = 1;
    int column = 1;

    if (json.empty())
    {
        throw_serialization_error("Invalid JSON format: empty JSON string.", 0, 0);
    }

    for (size_t index = 0; index < json.size(); index++)
    {
        const char c = json[index];

        if (is_whitespace(c))
        {
            SC_DECL_CACULATE_LINE_COLUMN(c)
            continue;
        }

        switch (state)
        {
        case JsonState::Start:
            if (index > 0 || c != '[')
            {
                throw_serialization_error("Invalid JSON format: missing opening bracket.", line, column);
            }

            state = JsonState::ObjectValue;
            break; /* end case JsonState::Start */
        case JsonState::ObjectValue:
            value_expression.clear();

            if (c == ']')
            {
                if (after_comma)
                {
                    throw_serialization_error("Invalid JSON format: trailing comma before closing bracket.", line, column);
                }
                // Empty array.
                state = JsonState::End;
                break;
            }

            after_comma = false;

            if (c == '{')
            {
                // go to the end of the object (string-aware so braces inside
                // string values do not unbalance the scan)
                scan_balanced_expression(json, index, '{', '}', line, column, value_expression);

                // NOLINTNEXTLINE(misc-no-recursion)
                json_array.emplace_back(std::make_unique<JsonNode>(JsonType::Object, value_expression));
            }
            else if (c == '[')
            {
                // array in array is valid in JSON, but we do not support it yet
                throw_serialization_error("Invalid JSON format: unexpected array in array.", line, column);
            }
            else
            {
                if (!std::isalpha(c) && !is_json_number(c) && c != '"') // check if the value is a string, number, boolean, or null
                {
                    throw_serialization_error(std::format("Invalid JSON format: invalid value expression: {}",
                                                          value_expression),
                                              line,
                                              column);
                }

                // we assume the value is a string, number, boolean, or null, and the expression is correct
                // if the expression is not correct, the exception will be thrown when parsing the value later
                // Track quoted-string state so that ',' or ']' inside a
                // string value does not terminate the scan early.
                bool in_string = false;
                bool escape = false;
                while (index < json.size() - 1)
                {
                    const char c_curr = json[index];
                    value_expression.push_back(c_curr);
                    if (escape)
                    {
                        escape = false;
                    }
                    else if (in_string)
                    {
                        if (c_curr == '\\')
                        {
                            escape = true;
                        }
                        else if (c_curr == '"')
                        {
                            in_string = false;
                        }
                    }
                    else if (c_curr == '"')
                    {
                        in_string = true;
                    }

                    const char c_next = json[index + 1];
                    if (!in_string && (c_next == ',' || c_next == ']'))
                    {
                        break;
                    }

                    SC_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }

                value_expression = trim_string(value_expression);

                JsonType type = JsonType::Unknown;
                if (value_expression == "true" || value_expression == "false")
                {
                    type = JsonType::Boolean;
                }
                else if (value_expression == "null")
                {
                    type = JsonType::Null;
                }
                else if (value_expression[0] == '"' && value_expression[value_expression.size() - 1] == '"')
                {
                    type = JsonType::String;
                }
                else
                {
                    bool is_number = true;
                    for (const char c : value_expression)
                    {
                        if (!is_json_number(c))
                        {
                            is_number = false;
                            break;
                        }
                    }
                    if (is_number)
                    {
                        type = JsonType::Number;
                    }
                    else
                    {
                        throw_serialization_error(std::format("Invalid JSON format: invalid value expression: {}",
                                                              value_expression),
                                                  line,
                                                  column);
                    }
                }

                json_array.emplace_back(std::make_unique<JsonNode>(type, value_expression));
            }

            state = JsonState::Comma;

            break; /* end case JsonState::ObjectValue */
        case JsonState::Comma:
            if (c == ',')
            {
                state = JsonState::ObjectValue;
                after_comma = true;
            }
            else if (c == ']')
            {
                state = JsonState::End;
            }
            break; /* end case JsonState::Comma */
        case JsonState::End:
            if (index != json.size() - 1)
            {
                throw_serialization_error("Invalid JSON format: extra characters after closing bracket.", line, column);
            }
            break; /* end case JsonState::End */
        case JsonState::ObjectKey:
            throw_serialization_error("Invalid JSON format: unexpected key in array.", line, column);
            break; /* end case JsonState::ObjectKey */
        case JsonState::Column:
            throw_serialization_error("Invalid JSON format: unexpected column in array.", line, column);
            break; /* end case JsonState::Column */
        }
    }

    if (state != JsonState::End)
    {
        throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
    }

    return json_array;
}
// NOLINTEND(readability-function-cognitive-complexity)

// NOLINTBEGIN(readability-function-cognitive-complexity)
// NOLINTNEXTLINE(misc-no-recursion)
JsonMap JsonNode::parse_object(const std::string & json)
{
    JsonMap json_map;
    JsonState state = JsonState::Start;
    std::string key;
    std::string value_expression;
    bool after_comma = false;

    int line = 1;
    int column = 1;

    if (json.empty())
    {
        throw_serialization_error("Invalid JSON format: empty JSON string.", 0, 0);
    }

    for (size_t index = 0; index < json.size(); index++)
    {
        const char c = json[index];

        if (is_whitespace(c))
        {
            SC_DECL_CACULATE_LINE_COLUMN(c)
            continue;
        }

        switch (state)
        {
        case JsonState::Start:
            if (index > 0 || c != '{')
            {
                throw_serialization_error("Invalid JSON format: missing opening bracket.", line, column);
            }

            state = JsonState::ObjectKey;
            break; /* end case JsonState::Start */
        case JsonState::ObjectKey:
            if (c == '}')
            {
                if (after_comma)
                {
                    throw_serialization_error("Invalid JSON format: trailing comma before closing bracket.", line, column);
                }
                // Empty object.
                state = JsonState::End;
                break;
            }

            after_comma = false;

            if (c == '"')
            {
                key.clear();
                bool close = false; // get the key string directly

                index += 1;
                while (index < json.size())
                {
                    if (json[index] == '"')
                    {
                        close = true;
                        break;
                    }
                    key.push_back(json[index]);
                    SC_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }
                if (!close)
                {
                    throw_serialization_error("Invalid JSON format: missing closing quote for key.", line, column);
                }

                key = trim_string(key);
                state = JsonState::Column;
            }
            else
            {
                throw_serialization_error("Invalid JSON format: missing opening quote for key.", line, column);
            }
            break; /* end case JsonState::ObjectKey */
        case JsonState::Column:
            if (c == ':')
            {
                state = JsonState::ObjectValue;
            }
            break; /* end case JsonState::Column */
        case JsonState::ObjectValue:
            value_expression.clear();

            if (c == '{')
            {
                // go to the end of the object (string-aware so braces inside
                // string values do not unbalance the scan)
                scan_balanced_expression(json, index, '{', '}', line, column, value_expression);

                // NOLINTNEXTLINE(misc-no-recursion)
                json_map.emplace(key, std::make_unique<JsonNode>(JsonType::Object, value_expression));
            }
            else if (c == '[')
            {
                // go to the end of the array (string-aware so brackets inside
                // string values do not unbalance the scan)
                scan_balanced_expression(json, index, '[', ']', line, column, value_expression);

                // NOLINTNEXTLINE(misc-no-recursion)
                json_map.emplace(key, std::make_unique<JsonNode>(JsonType::Array, value_expression));
            }
            else
            {
                if (!std::isalpha(c) && !is_json_number(c) && c != '"') // check if the value is a string, number, boolean, or null
                {
                    throw_serialization_error(std::format("Invalid JSON format: invalid value expression: {}",
                                                          value_expression),
                                              line,
                                              column);
                }

                // we assume the value is a string, number, boolean, or null, and the expression is correct
                // if the expression is not correct, the exception will be thrown when parsing the value later
                // Track quoted-string state so that ',' or '}' inside a
                // string value does not terminate the scan early.
                bool in_string = false;
                bool escape = false;
                while (index < json.size() - 1)
                {
                    const char c_curr = json[index];
                    value_expression.push_back(c_curr);
                    if (escape)
                    {
                        escape = false;
                    }
                    else if (in_string)
                    {
                        if (c_curr == '\\')
                        {
                            escape = true;
                        }
                        else if (c_curr == '"')
                        {
                            in_string = false;
                        }
                    }
                    else if (c_curr == '"')
                    {
                        in_string = true;
                    }

                    const char c_next = json[index + 1];
                    if (!in_string && (c_next == ',' || c_next == '}'))
                    {
                        break;
                    }

                    SC_DECL_CACULATE_LINE_COLUMN(json[index])
                    index += 1;
                }

                value_expression = trim_string(value_expression);

                JsonType type = JsonType::Unknown;
                if (value_expression == "true" || value_expression == "false")
                {
                    type = JsonType::Boolean;
                }
                else if (value_expression == "null")
                {
                    type = JsonType::Null;
                }
                else if (value_expression[0] == '"' && value_expression[value_expression.size() - 1] == '"')
                {
                    type = JsonType::String;
                }
                else
                {
                    bool is_number = true;
                    for (const char c : value_expression)
                    {
                        if (!is_json_number(c))
                        {
                            is_number = false;
                            break;
                        }
                    }
                    if (is_number)
                    {
                        type = JsonType::Number;
                    }
                    else
                    {
                        throw_serialization_error(
                            std::format("Invalid JSON format: invalid value expression: {}", value_expression),
                            line,
                            column);
                    }
                }

                json_map.emplace(key, std::make_unique<JsonNode>(type, value_expression));
            }

            state = JsonState::Comma;

            break; /* end case JsonState::ObjectValue */
        case JsonState::Comma:
            if (c == ',')
            {
                state = JsonState::ObjectKey;
                after_comma = true;
            }
            else if (c == '}')
            {
                state = JsonState::End;
            }
            break; /* end case JsonState::Comma */
        case JsonState::End:
            if (index != json.size() - 1)
            {
                throw_serialization_error("Invalid JSON format: extra characters after closing bracket.", line, column);
            }
            break; /* end case JsonState::End */
        }
    }

    if (state != JsonState::End)
    {
        throw_serialization_error("Invalid JSON format: missing closing bracket.", line, column);
    }

    return json_map;
}
// NOLINTEND(readability-function-cognitive-complexity)

#undef SC_DECL_CACULATE_LINE_COLUMN

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
