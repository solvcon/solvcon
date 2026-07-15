#include <gtest/gtest.h>
#include <thread>
#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#include <solvcon/serialization/SerializableItem.hpp>
namespace solvcon
{

namespace detail
{

struct Address : SerializableItem
{
    std::string country;
    std::string city;
    std::vector<std::string> phone_numbers;
    std::vector<int> zip_codes;

    MM_DECL_SERIALIZABLE(
        register_member("country", country);
        register_member("city", city);
        register_member("phone_numbers", phone_numbers);
        register_member("zip_codes", zip_codes);)
}; /* end struct Address */

struct Pet : SerializableItem
{
    std::string name;
    int age;
    bool is_dog;
    bool is_cat;

    MM_DECL_SERIALIZABLE(
        register_member("name", name);
        register_member("age", age);
        register_member("is_dog", is_dog);
        register_member("is_cat", is_cat);)
}; /* end struct Pet */

struct Person : SerializableItem
{
    std::string name;
    int age;
    bool is_student;
    Address address;
    std::vector<Pet> pets;

    MM_DECL_SERIALIZABLE(
        register_member("name", name);
        register_member("age", age);
        register_member("is_student", is_student);
        register_member("address", address);
        register_member("pets", pets);)
}; /* end struct Person */

Pet create_dog()
{
    Pet pet;
    pet.name = "Fluffy";
    pet.age = 3;
    pet.is_dog = true;
    pet.is_cat = false;
    return pet;
}

Pet create_cat()
{
    Pet pet;
    pet.name = "Whiskers";
    pet.age = 8;
    pet.is_dog = false;
    pet.is_cat = true;
    return pet;
}

Address create_address()
{
    Address address;
    address.country = "USA";
    address.city = "New York";
    address.phone_numbers = {"123-456-7890", "098-765-4321"};
    address.zip_codes = {10001, 10002};
    return address;
}

struct SecretItem : SerializableItem
{
private:
    std::string private_info = "private_info";

public:
    std::string public_info = "public_info";

    std::string get_private_info() const
    {
        return private_info;
    }

    MM_DECL_SERIALIZABLE(
        /* not expose public_info */
        register_member("private_info", private_info);)
}; /* end struct SecretItem */

struct EscapeItem : SerializableItem
{
    std::string escape_string = "\"\\/\b\f\n\r\t";

    MM_DECL_SERIALIZABLE(
        register_member("escape_string", escape_string);)
}; /* end struct EscapeItem */

struct TestUnorderedMapItem : SerializableItem
{
    std::unordered_map<std::string, int> numer_map;
    std::unordered_map<std::string, Pet> pet_map;

    MM_DECL_SERIALIZABLE(
        register_member("numer_map", numer_map);
        register_member("pet_map", pet_map);)
}; /* end struct TestUnorderedMapItem */

} /* end namespace detail */

TEST(Json, serialize_private_member_partial_exposure)
{
    detail::SecretItem secret;
    std::string json = secret.to_json();
    EXPECT_EQ(json, "{\"private_info\":\"private_info\"}");
}

TEST(Json, serialize_escape)
{
    detail::EscapeItem escape;
    std::string json = escape.to_json();
    EXPECT_EQ(json, "{\"escape_string\":\"\\\"\\\\/\\b\\f\\n\\r\\t\"}");
}

TEST(Json, serialize_simple)
{
    auto pet = detail::create_dog();
    std::string json = pet.to_json();
    EXPECT_EQ(json, "{\"name\":\"Fluffy\",\"age\":3,\"is_dog\":true,\"is_cat\":false}");
}

TEST(Json, serialize_wrong_order)
{
    auto pet = detail::create_dog();
    std::string json = pet.to_json();
    EXPECT_NE(json, "{\"name\":\"Fluffy\",\"age\":3,\"is_cat\":false,\"is_dog\":true\"}"); // cat and dog are swapped
}

TEST(Json, serialize_with_vector)
{
    auto address = detail::create_address();

    std::string json = address.to_json();
    EXPECT_EQ(json, "{\"country\":\"USA\",\"city\":\"New York\",\"phone_numbers\":[\"123-456-7890\",\"098-765-4321\"],\"zip_codes\":[10001,10002]}");
}

TEST(Json, serialize_with_object)
{

    detail::Person person;
    person.name = "John Doe";
    person.age = 30;
    person.is_student = true;
    person.address = detail::create_address();
    person.pets.push_back(detail::create_dog());
    person.pets.push_back(detail::create_cat());

    std::string json = person.to_json();

    std::string answer = std::string("{\"name\":\"John Doe\",\"age\":30,\"is_student\":true,") +
                         "\"address\":{\"country\":\"USA\",\"city\":\"New York\",\"phone_numbers\":[\"123-456-7890\",\"098-765-4321\"],\"zip_codes\":[10001,10002]}," +
                         "\"pets\":[{\"name\":\"Fluffy\",\"age\":3,\"is_dog\":true,\"is_cat\":false},{\"name\":\"Whiskers\",\"age\":8,\"is_dog\":false,\"is_cat\":true}]}";
}

TEST(Json, serialize_with_unordered_map)
{
    detail::TestUnorderedMapItem item;
    item.numer_map["one"] = 1;
    item.numer_map["two"] = 2;
    item.numer_map["three"] = 3;
    item.pet_map["dog"] = detail::create_dog();
    item.pet_map["cat"] = detail::create_cat();

    std::string json = item.to_json();

    // the key of numer_map is sorted
    EXPECT_EQ(json, "{\"numer_map\":{\"one\":1,\"three\":3,\"two\":2},\"pet_map\":{\"cat\":{\"name\":\"Whiskers\",\"age\":8,\"is_dog\":false,\"is_cat\":true},\"dog\":{\"name\":\"Fluffy\",\"age\":3,\"is_dog\":true,\"is_cat\":false}}}");
}

TEST(Json, deserialize_private_member_partial_exposure)
{
    std::string json = "{\"private_info\":\"private_info\"}";
    detail::SecretItem secret;
    secret.from_json(json);
    EXPECT_EQ(secret.get_private_info(), "private_info");
}

TEST(json, deserialize_simple)
{
    std::string json = "{\"name\":\"Fluffy\",\"age\":3,\"is_dog\":true,\"is_cat\":false}";
    detail::Pet pet;
    pet.from_json(json);
    EXPECT_EQ(pet.name, "Fluffy");
    EXPECT_EQ(pet.age, 3);
    EXPECT_EQ(pet.is_dog, true);
    EXPECT_EQ(pet.is_cat, false);
}

TEST(json, deserialize_trim)
{
    std::string json = "{  \n \"name\" \t\t\t \t\n:\"Fluffy\", \t\n\"age\":3,\"is_dog\": \n\ttrue,   \"is_cat\":false\t\t}";
    detail::Pet pet;
    pet.from_json(json);
    EXPECT_EQ(pet.name, "Fluffy");
    EXPECT_EQ(pet.age, 3);
    EXPECT_EQ(pet.is_dog, true);
    EXPECT_EQ(pet.is_cat, false);
}

TEST(json, deserialize_with_vector)
{
    std::string json = "{\"country\":\"USA\",\"city\":\"New York\",\"phone_numbers\":[\"123-456-7890\",\"098-765-4321\"],\"zip_codes\":[10001,10002]}";
    detail::Address address;
    address.from_json(json);
    EXPECT_EQ(address.country, "USA");
    EXPECT_EQ(address.city, "New York");
    EXPECT_EQ(address.phone_numbers.size(), 2);
    EXPECT_EQ(address.phone_numbers[0], "123-456-7890");
    EXPECT_EQ(address.phone_numbers[1], "098-765-4321");
    EXPECT_EQ(address.zip_codes.size(), 2);
    EXPECT_EQ(address.zip_codes[0], 10001);
    EXPECT_EQ(address.zip_codes[1], 10002);
}

TEST(json, deserialize_with_object)
{
    std::string json = R"({
        "name": "John Doe",
        "age": 30,
        "is_student": true,
        "address": {
            "country": "USA",
            "city": "New York",
            "phone_numbers": [
                "123-456-7890",
                "098-765-4321"
            ],
            "zip_codes": [
                10001,
                10002
            ]
        },
        "pets": [
            {
                "name": "Fluffy",
                "age": 3,
                "is_dog": true,
                "is_cat": false
            },
            {
                "name": "Whiskers",
                "age": 8,
                "is_dog": false,
                "is_cat": true
            }]
        }
    })";

    detail::Person person;
    person.from_json(json);

    EXPECT_EQ(person.name, "John Doe");
    EXPECT_EQ(person.age, 30);
    EXPECT_EQ(person.is_student, true);

    EXPECT_EQ(person.address.country, "USA");
    EXPECT_EQ(person.address.city, "New York");
    EXPECT_EQ(person.address.phone_numbers.size(), 2);
    EXPECT_EQ(person.address.phone_numbers[0], "123-456-7890");
    EXPECT_EQ(person.address.phone_numbers[1], "098-765-4321");
    EXPECT_EQ(person.address.zip_codes.size(), 2);
    EXPECT_EQ(person.address.zip_codes[0], 10001);
    EXPECT_EQ(person.address.zip_codes[1], 10002);

    EXPECT_EQ(person.pets.size(), 2);
    EXPECT_EQ(person.pets[0].name, "Fluffy");
    EXPECT_EQ(person.pets[0].age, 3);
    EXPECT_EQ(person.pets[0].is_dog, true);
    EXPECT_EQ(person.pets[0].is_cat, false);

    EXPECT_EQ(person.pets[1].name, "Whiskers");
    EXPECT_EQ(person.pets[1].age, 8);
    EXPECT_EQ(person.pets[1].is_dog, false);
    EXPECT_EQ(person.pets[1].is_cat, true);
}

TEST(Json, deserialize_with_unordered_map)
{
    std::string json = "{\"numer_map\":{\"one\":1,\"three\":3,\"two\":2},\"pet_map\":{\"cat\":{\"name\":\"Whiskers\",\"age\":8,\"is_dog\":false,\"is_cat\":true},\"dog\":{\"name\":\"Fluffy\",\"age\":3,\"is_dog\":true,\"is_cat\":false}}}";
    detail::TestUnorderedMapItem item;
    item.from_json(json);

    EXPECT_EQ(item.numer_map.size(), 3);
    EXPECT_EQ(item.numer_map["one"], 1);
    EXPECT_EQ(item.numer_map["two"], 2);
    EXPECT_EQ(item.numer_map["three"], 3);

    EXPECT_EQ(item.pet_map.size(), 2);
    EXPECT_EQ(item.pet_map["dog"].name, "Fluffy");
    EXPECT_EQ(item.pet_map["dog"].age, 3);
    EXPECT_EQ(item.pet_map["dog"].is_dog, true);
    EXPECT_EQ(item.pet_map["dog"].is_cat, false);

    EXPECT_EQ(item.pet_map["cat"].name, "Whiskers");
    EXPECT_EQ(item.pet_map["cat"].age, 8);
    EXPECT_EQ(item.pet_map["cat"].is_dog, false);
    EXPECT_EQ(item.pet_map["cat"].is_cat, true);
}

// The tests below cover empty containers, lenient trailing commas, and
// string values that embed JSON delimiters (',', ']', '}') or escape
// sequences. They exercise the parser directly through detail::JsonNode so
// that the produced structure can be inspected precisely.

TEST(Json, parse_empty_array)
{
    detail::JsonNode node(detail::JsonType::Array, "[]");
    auto & arr = std::get<detail::JsonArray>(node.value);
    EXPECT_EQ(arr.size(), 0);
}

TEST(Json, parse_empty_object)
{
    detail::JsonNode node(detail::JsonType::Object, "{}");
    auto & map = std::get<detail::JsonMap>(node.value);
    EXPECT_EQ(map.size(), 0);
}

TEST(Json, parse_empty_array_and_object_with_whitespace)
{
    detail::JsonNode array_node(detail::JsonType::Array, "[   ]");
    EXPECT_EQ(std::get<detail::JsonArray>(array_node.value).size(), 0);

    detail::JsonNode object_node(detail::JsonType::Object, "{ \n\t }");
    EXPECT_EQ(std::get<detail::JsonMap>(object_node.value).size(), 0);
}

TEST(Json, parse_nested_empty_array_and_object)
{
    detail::JsonNode node(detail::JsonType::Object, "{\"arr\":[],\"obj\":{}}");
    auto & map = std::get<detail::JsonMap>(node.value);
    ASSERT_EQ(map.size(), 2);
    EXPECT_EQ(map.at("arr")->type, detail::JsonType::Array);
    EXPECT_EQ(std::get<detail::JsonArray>(map.at("arr")->value).size(), 0);
    EXPECT_EQ(map.at("obj")->type, detail::JsonType::Object);
    EXPECT_EQ(std::get<detail::JsonMap>(map.at("obj")->value).size(), 0);
}

TEST(Json, parse_array_string_values_with_delimiters)
{
    detail::JsonNode node(detail::JsonType::Array, "[\"a,b\",\"c]d\",\"e{f}g\"]");
    auto & arr = std::get<detail::JsonArray>(node.value);
    ASSERT_EQ(arr.size(), 3);
    EXPECT_EQ(arr[0]->type, detail::JsonType::String);
    EXPECT_EQ(std::get<std::string>(arr[0]->value), "\"a,b\"");
    EXPECT_EQ(std::get<std::string>(arr[1]->value), "\"c]d\"");
    EXPECT_EQ(std::get<std::string>(arr[2]->value), "\"e{f}g\"");
}

TEST(Json, parse_object_string_value_with_delimiters)
{
    detail::JsonNode node(detail::JsonType::Object, "{\"key\":\"a,b}c]d\"}");
    auto & map = std::get<detail::JsonMap>(node.value);
    ASSERT_EQ(map.size(), 1);
    EXPECT_EQ(map.at("key")->type, detail::JsonType::String);
    EXPECT_EQ(std::get<std::string>(map.at("key")->value), "\"a,b}c]d\"");
}

TEST(Json, parse_string_with_escaped_quote)
{
    // The escaped quote stays inside the string, so the following comma
    // must not terminate the value scan early.
    detail::JsonNode node(detail::JsonType::Array, "[\"a\\\",b\",\"c\"]");
    auto & arr = std::get<detail::JsonArray>(node.value);
    ASSERT_EQ(arr.size(), 2);
    EXPECT_EQ(arr[0]->type, detail::JsonType::String);
    EXPECT_EQ(std::get<std::string>(arr[0]->value), "\"a\\\",b\"");
    EXPECT_EQ(std::get<std::string>(arr[1]->value), "\"c\"");
}

TEST(Json, parse_string_with_escaped_backslash)
{
    // A trailing escaped backslash must leave the closing quote able to
    // terminate the string.
    detail::JsonNode node(detail::JsonType::Array, "[\"a\\\\\",\"b\"]");
    auto & arr = std::get<detail::JsonArray>(node.value);
    ASSERT_EQ(arr.size(), 2);
    EXPECT_EQ(std::get<std::string>(arr[0]->value), "\"a\\\\\"");
    EXPECT_EQ(std::get<std::string>(arr[1]->value), "\"b\"");
}

TEST(Json, parse_array_trailing_comma_rejected)
{
    // A trailing comma before the closing bracket is invalid JSON and is
    // rejected, while an empty array remains valid.
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Array, "[1,2,]"), std::runtime_error);
    EXPECT_NO_THROW(detail::JsonNode(detail::JsonType::Array, "[]"));
}

TEST(Json, parse_object_trailing_comma_rejected)
{
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Object, "{\"a\":1,}"), std::runtime_error);
    EXPECT_NO_THROW(detail::JsonNode(detail::JsonType::Object, "{}"));
}

TEST(Json, parse_nested_array_string_with_bracket)
{
    // A ']' or '[' embedded in a string inside a nested array must not end
    // the array extraction early.
    detail::JsonNode node(detail::JsonType::Object, "{\"k\":[\"a]b\",\"c[d\"]}");
    auto & map = std::get<detail::JsonMap>(node.value);
    ASSERT_EQ(map.size(), 1);
    EXPECT_EQ(map.at("k")->type, detail::JsonType::Array);
    auto & arr = std::get<detail::JsonArray>(map.at("k")->value);
    ASSERT_EQ(arr.size(), 2);
    EXPECT_EQ(std::get<std::string>(arr[0]->value), "\"a]b\"");
    EXPECT_EQ(std::get<std::string>(arr[1]->value), "\"c[d\"");
}

TEST(Json, parse_nested_object_string_with_brace)
{
    // A '}' or '{' embedded in a string inside a nested object must not end
    // the object extraction early.
    detail::JsonNode node(detail::JsonType::Object, "{\"k\":{\"x\":\"a}b{c\"}}");
    auto & map = std::get<detail::JsonMap>(node.value);
    ASSERT_EQ(map.size(), 1);
    EXPECT_EQ(map.at("k")->type, detail::JsonType::Object);
    auto & inner = std::get<detail::JsonMap>(map.at("k")->value);
    ASSERT_EQ(inner.size(), 1);
    EXPECT_EQ(std::get<std::string>(inner.at("x")->value), "\"a}b{c\"");
}

TEST(Json, parse_nested_container_string_with_escape)
{
    // The container scanner must honour escapes: the escaped quote keeps the
    // ']' inside the string, so the nested array is not closed early.
    detail::JsonNode node(detail::JsonType::Object, "{\"k\":[\"a\\\"]b\"]}");
    auto & map = std::get<detail::JsonMap>(node.value);
    ASSERT_EQ(map.size(), 1);
    EXPECT_EQ(map.at("k")->type, detail::JsonType::Array);
    auto & arr = std::get<detail::JsonArray>(map.at("k")->value);
    ASSERT_EQ(arr.size(), 1);
    EXPECT_EQ(std::get<std::string>(arr[0]->value), "\"a\\\"]b\"");
}

TEST(Json, parse_invalid_comma_sequences_rejected)
{
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Array, "[1, ]"), std::runtime_error);
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Array, "[1,,2]"), std::runtime_error);
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Array, "[,]"), std::runtime_error);
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Object, "{\"k\":[1,2,]}"), std::runtime_error);
    EXPECT_NO_THROW(detail::JsonNode(detail::JsonType::Array, "[1, 2]"));
}

TEST(Json, parse_unterminated_nested_container_rejected)
{
    // The container scanner must throw when a nested container or a string
    // inside it is never closed.
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Object, "{\"k\":[1,2"), std::runtime_error);
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Object, "{\"k\":{\"x\":1"), std::runtime_error);
    EXPECT_THROW(detail::JsonNode(detail::JsonType::Object, "{\"k\":[\"a"), std::runtime_error);
}

TEST(Json, deserialize_empty_vector)
{
    std::string json = "{\"country\":\"USA\",\"city\":\"New York\",\"phone_numbers\":[],\"zip_codes\":[]}";
    detail::Address address;
    address.from_json(json);
    EXPECT_EQ(address.country, "USA");
    EXPECT_EQ(address.city, "New York");
    EXPECT_EQ(address.phone_numbers.size(), 0);
    EXPECT_EQ(address.zip_codes.size(), 0);
}

TEST(Json, deserialize_empty_unordered_map)
{
    std::string json = "{\"numer_map\":{},\"pet_map\":{}}";
    detail::TestUnorderedMapItem item;
    item.from_json(json);
    EXPECT_EQ(item.numer_map.size(), 0);
    EXPECT_EQ(item.pet_map.size(), 0);
}

TEST(Json, deserialize_string_with_delimiters)
{
    std::string json = "{\"country\":\"a]b\",\"city\":\"New York, NY\",\"phone_numbers\":[\"1,2\",\"3}4\"],\"zip_codes\":[]}";
    detail::Address address;
    address.from_json(json);
    EXPECT_EQ(address.country, "a]b");
    EXPECT_EQ(address.city, "New York, NY");
    ASSERT_EQ(address.phone_numbers.size(), 2);
    EXPECT_EQ(address.phone_numbers[0], "1,2");
    EXPECT_EQ(address.phone_numbers[1], "3}4");
    EXPECT_EQ(address.zip_codes.size(), 0);
}

TEST(Json, deserialize_nested_string_with_bracket)
{
    // A ']' inside a string element of a nested array must survive the
    // object-level container extraction (string-aware depth scan).
    std::string json = "{\"country\":\"USA\",\"city\":\"NYC\",\"phone_numbers\":[\"1]2\",\"3\"],\"zip_codes\":[]}";
    detail::Address address;
    address.from_json(json);
    EXPECT_EQ(address.country, "USA");
    ASSERT_EQ(address.phone_numbers.size(), 2);
    EXPECT_EQ(address.phone_numbers[0], "1]2");
    EXPECT_EQ(address.phone_numbers[1], "3");
    EXPECT_EQ(address.zip_codes.size(), 0);
}

TEST(Json, round_trip_string_with_comma)
{
    detail::Address address;
    address.country = "USA";
    address.city = "New York, NY";
    address.phone_numbers = {"1,2", "3"};
    address.zip_codes = {};

    detail::Address restored;
    restored.from_json(address.to_json());
    EXPECT_EQ(restored.country, address.country);
    EXPECT_EQ(restored.city, address.city);
    ASSERT_EQ(restored.phone_numbers.size(), 2);
    EXPECT_EQ(restored.phone_numbers[0], "1,2");
    EXPECT_EQ(restored.phone_numbers[1], "3");
    EXPECT_EQ(restored.zip_codes.size(), 0);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
