rm -rf ./data
mkdir /opt/data/

wget -P /opt/data/ https://www.dropbox.com/s/802qjuws26wp0ww/data-rev.zip
unzip -o /opt/data/data-rev.zip -d /opt/data/
rm /opt/data/data-rev.zip

wget -P /opt/data/ https://www.dropbox.com/s/g0bqpgstzv2ojv7/data.zip
unzip -o /opt/data/data.zip -d /opt/data/
rm /opt/data/data.zip

wget -P /opt/data/ https://www.dropbox.com/s/tz37aitrl2umibu/data-normal1.zip
unzip -o /opt/data/data-normal1.zip -d /opt/data/
rm /opt/data/data-normal1.zip

wget -P /opt/data/ https://www.dropbox.com/s/6c45r42p6imoeqv/data-normal-rev1.zip
unzip -o /opt/data/data-normal-rev1.zip -d /opt/data/
rm /opt/data/data-normal-rev1.zip

wget -P /opt/data/ https://www.dropbox.com/s/o27x0nfu2klndd0/data-left.zip
unzip -o /opt/data/data-left.zip -d /opt/data/
rm /opt/data/data-left.zip