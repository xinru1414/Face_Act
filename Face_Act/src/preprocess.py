"""
Jan 2020
Xinru Yan

This program prepares the donation dialogue data ready

Input:
    data_annotate: 297 dialogues annotated with face labels
    data_info: 1,018 dialogue information on donation
    data_full: 1,018 dialogues
Output:
    a csv file with each row containing the 'message_id', 'text' of full conversation and the 'donation_label' (0 for no
    donation, 1 for the persuadee made a donation)
"""
import click
import csv
from enum import Enum


class DataInfoHeader(Enum):
    CONVERSATION_ID = 'B2'
    DONATION = 'B6'

    @property
    def column_name(self):
        return self.value


class DataFaceHeader(Enum):
    CONVERSATION_ID = 'B2'
    SENT = 'Unit'
    FACE_LABEL = 'Poss_labels'

    @property
    def column_name(self):
        return self.value


class Conversation:
    def __init__(self, message_id):
        self.message_id = message_id
        self.sents = []
        self.label = ''

    def add_sent(self, sent):
        self.sents += [sent]

    def add_label(self, label):
        self.label = label

    def __len__(self):
        return len(self.sents)

    @property
    def dict(self):
        d = {'message_id': self.message_id,
             'text': ' '.join(self.sents),
             'donation_label': self.label}
        return d


def read_data(csv_file, conversations):
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[DataFaceHeader.CONVERSATION_ID.column_name]
            if message_id not in conversations:
                conversations[message_id] = Conversation(message_id)
            sent = row[DataFaceHeader.SENT.column_name]
            conversations[message_id].add_sent(sent)
    return conversations


def read_face_label(csv_file):
    conversations = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[DataFaceHeader.CONVERSATION_ID.column_name]
            if message_id not in conversations:
                conversations[message_id] = Conversation(message_id)
            face_label = row[DataFaceHeader.FACE_LABEL.column_name]
            if len(face_label) > 6:
                face_label = face_label[:5]
            if face_label == '':
                face_label = 'None'
            conversations[message_id].add_sent(face_label)
    return conversations


def write_data(csv_file, conversations):
    with open(csv_file, 'w') as csvfile:
        headers = ['message_id', 'text', 'donation_label']
        writer = csv.DictWriter(csvfile, headers, extrasaction='ignore')
        writer.writeheader()
        for conversation in conversations.values():
            writer.writerow(conversation.dict)


def donation(value: str):
    if float(value) > 0:
        label = 1
    else:
        label = 0
    return label


@click.command()
@click.option('-i', '--dataset_info', default='../data/data_info.csv')
@click.option('-f', '--dataset_full', default='../data/data_full.csv')
@click.option('-a', '--dataset_annotate', default='../data/data_annotate.csv')
@click.option('-o_e', '--output_entire', default='../data/1017_prepared_data.csv')
@click.option('-o_a', '--output_annotate', default='../data/296_prepared_data.csv')
@click.option('-o_c', '--output_facelabel', default='../data/296_face_label_prepared_data.csv')
def main(dataset_info, dataset_full, dataset_annotate, output_entire, output_annotate, output_facelabel):
    total_conversations = {}
    total_conversations = read_data(dataset_annotate, total_conversations)
    total_conversations = read_data(dataset_full, total_conversations)

    annotate_conversations = {}
    annotate_conversations = read_data(dataset_annotate, annotate_conversations)

    face_sequences = read_face_label(dataset_annotate)

    print(f'annotated dataset {len(annotate_conversations)}, total dataset {len(total_conversations)}')

    with open(f'{dataset_info}', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[DataInfoHeader.CONVERSATION_ID.column_name]
            if message_id not in total_conversations:
                total_conversations[message_id] = Conversation(message_id)
            if message_id in total_conversations:
                value = row[DataInfoHeader.DONATION.column_name]
                label = donation(value)
                total_conversations[message_id].add_label(label)
            if message_id in annotate_conversations:
                value = row[DataInfoHeader.DONATION.column_name]
                label = donation(value)
                annotate_conversations[message_id].add_label(label)
            if message_id in face_sequences:
                value = row[DataInfoHeader.DONATION.column_name]
                label = donation(value)
                face_sequences[message_id].add_label(label)

    write_data(output_entire, total_conversations)
    write_data(output_annotate, annotate_conversations)
    write_data(output_facelabel, face_sequences)


if __name__ == '__main__':
    main()