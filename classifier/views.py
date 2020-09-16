from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import  APIView
from rest_framework.response import  Response
from rest_framework.permissions import AllowAny

from .serializers import TweetSerializer
from .utils import  PolarityClassifier

path_to_model = '/media/sajjad/New Volume/Projects/sentiment_analysis_django/sentiment_analysis/classifier/deep_models/model_lstm_cnn.h5'
path_to_tokenizer = '/media/sajjad/New Volume/Projects/sentiment_analysis_django/sentiment_analysis/classifier/deep_models/tokenizer.pkl'
classifier = PolarityClassifier(path_to_model=path_to_model,path_to_tokenizer=path_to_tokenizer)


class Predict(APIView) :
    permission_classes = (AllowAny,)

    def get (self,request,*args,**kwargs) :
        serializer = TweetSerializer(data=request.data,many=True)
        if serializer.is_valid(raise_exception=True) :
            tweet = serializer.save()
            texts = [element.text for element in tweet]
            y_pred = classifier.predict(texts)
            data = dict(zip(texts,y_pred))
            return Response(data=data,status=status.HTTP_200_OK)
        return Response(data=serializer.errors,status=status.HTTP_400_BAD_REQUEST)
