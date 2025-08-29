import transformers
import torch
from preprocess import preprocess

tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

assert tokenizer.bos_token_id == None
assert tokenizer.eos_token_id != None
bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
assert bos_token_id == tokenizer.eos_token_id


mock_data = {
    "train": [
        {"en": "A small dog is playing.", "de": "Ein kleiner Hund spielt."},
        {"en": "The sun is shining.", "de": "Die Sonne scheint."},
        {"en": "She is making a cup of coffee.", "de": "Sie macht eine Tasse Kaffee."},
        {
            "en": "We need three apples and two bananas.",
            "de": "Wir brauchen drei Äpfel und zwei Bananen.",
        },
        {
            "en": "The bus arrives at 8 o'clock every morning.",
            "de": "Der Bus kommt jeden Morgen um 8 Uhr an.",
        },
        {
            "en": "He rides his bicycle to work.",
            "de": "Er fährt mit seinem Fahrrad zur Arbeit.",
        },
        {"en": "Leaves turn yellow in autumn.", "de": "Blätter werden im Herbst gelb."},
        {"en": "It is raining heavily outside.", "de": "Draußen regnet es stark."},
        {"en": "What time is it now?", "de": "Wie spät ist es jetzt?"},
        {"en": "My favorite color is blue.", "de": "Meine Lieblingsfarbe ist blau."},
    ],
    "validation": [
        {"en": "A cat is sleeping.", "de": "Eine Katze schläft."},
        {
            "en": "They are watching a movie together.",
            "de": "Sie schauen zusammen einen Film.",
        },
        {
            "en": "This book is very interesting.",
            "de": "Dieses Buch ist sehr interessant.",
        },
        {
            "en": "The park is crowded on weekends.",
            "de": "Der Park ist an Wochenenden belebt.",
        },
        {
            "en": "I wash my hands before eating.",
            "de": "Ich wasche mir die Hände vor dem Essen.",
        },
    ],
    "test": [
        {"en": "A bird is flying.", "de": "Ein Vogel fliegt."},
        {
            "en": "She told me she would visit her grandparents next week.",
            "de": "Sie sagte mir, dass sie nächste Woche ihre Großeltern besuchen würde.",
        },
        {
            "en": "The book that I bought yesterday is about history.",
            "de": "Das Buch, das ich gestern gekauft habe, handelt von Geschichte.",
        },
        {
            "en": "The shop will close early today because of the storm.",
            "de": "Der Laden schließt heute wegen des Sturms früh.",
        },
        {
            "en": "How much does this red dress cost?",
            "de": "Wie viel kostet dieses rote Kleid?",
        },
        {
            "en": "People usually give gifts during Christmas.",
            "de": "Leute schenken normalerweise zu Weihnachten Geschenke.",
        },
    ],
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, valid_dataloader, test_dataloader, data_test = preprocess(
    tokenizer, device=device, batch_size=2, dataset=mock_data
)
assert len(train_dataloader) == 5
assert len(valid_dataloader) == 3
assert len(test_dataloader) == 3

src, tgt, src_key_padding_mask, tgt_key_padding_mask = next(iter(valid_dataloader))
bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
eos_token_id = tokenizer.eos_token_id

print(src.shape)
print(tgt.shape)
assert src.shape == (2, 8)
assert tgt.shape == (2, 14)
assert src_key_padding_mask.shape == (2, 8)
assert tgt_key_padding_mask.shape == (2, 14)
assert torch.equal(src[:, -1], torch.tensor([eos_token_id, eos_token_id]))
assert torch.equal(tgt[:, 0], torch.tensor([bos_token_id, bos_token_id]))
assert torch.equal(tgt[:, -1], torch.tensor([eos_token_id, eos_token_id]))
assert torch.equal(
    src_key_padding_mask,
    torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        device=device,
    ),
)
